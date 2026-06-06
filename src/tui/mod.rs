use std::io::{self, Stdout};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use sysinfo::{CpuRefreshKind, RefreshKind, System};
use tokio_util::sync::CancellationToken;

use crate::metrics::SharedMetrics;
use log::error;

pub fn run(metrics: SharedMetrics, cancel: CancellationToken) {
    if let Err(e) = run_inner(&metrics, &cancel) {
        error!("[tui] error: {e}");
    }
    cancel.cancel();
}

fn run_inner(metrics: &SharedMetrics, cancel: &CancellationToken) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = render_loop(&mut terminal, metrics, cancel);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn render_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    metrics: &SharedMetrics,
    cancel: &CancellationToken,
) -> io::Result<()> {
    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything()));
    // First call is a baseline; the second (after the minimum interval) gives real deltas.
    sys.refresh_cpu_all();
    std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
    sys.refresh_cpu_all();

    let mut cpu_usages: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
    let mut last_cpu_refresh = Instant::now();

    loop {
        if cancel.is_cancelled() {
            break;
        }

        if last_cpu_refresh.elapsed() >= Duration::from_millis(500) {
            sys.refresh_cpu_all();
            for (i, cpu) in sys.cpus().iter().enumerate() {
                if i < cpu_usages.len() {
                    cpu_usages[i] = cpu.cpu_usage();
                }
            }
            last_cpu_refresh = Instant::now();
        }

        terminal.draw(|frame| draw(frame, metrics, &cpu_usages))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match (key.code, key.modifiers) {
                    (KeyCode::Char('q') | KeyCode::Char('Q'), _)
                    | (KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

fn draw(frame: &mut ratatui::Frame, metrics: &SharedMetrics, cpu_usages: &[f32]) {
    let area = frame.area();

    let [left, right] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(36), Constraint::Min(24)])
        .areas(area);

    draw_stats(frame, left, metrics);
    draw_cpu(frame, right, cpu_usages);
}

fn draw_stats(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, metrics: &SharedMetrics) {
    let rate = metrics.compute_rate();
    let session = metrics.session_shuffles.load(Ordering::Relaxed);
    let lifetime = metrics.lifetime_shuffles.load(Ordering::Relaxed);
    let last_best = metrics.last_report_best.load(Ordering::Relaxed);
    let ses_best = metrics.session_best.load(Ordering::Relaxed);
    let all_best = metrics.all_time_best.load(Ordering::Relaxed);
    let status = metrics.status.lock().clone();
    let uptime = fmt_uptime(metrics.started_at.elapsed().as_secs());
    let sparkline = {
        let history = metrics.recent_bests.lock();
        history
            .iter()
            .rev()
            .take(10)
            .map(|&v| best_char(v))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>()
    };

    let highlight = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let value = Style::default().fg(Color::White);

    let lines = vec![
        row("local rate", fmt_rate(rate), highlight),
        Line::raw(""),
        row("last 1s best", fmt_best(last_best), value),
        row("session best", fmt_best(ses_best), value),
        row("all-time best", fmt_best(all_best), value),
        Line::raw(""),
        row(
            "last 10 best",
            if sparkline.is_empty() {
                "—".into()
            } else {
                sparkline
            },
            value,
        ),
        Line::raw(""),
        row("this session", fmt_count(session), value),
        row("lifetime", fmt_count(lifetime), value),
        Line::raw(""),
        row("uptime", uptime, value),
        row("status", status, value),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" bogoforge ")
        .border_style(Style::default().fg(Color::DarkGray));

    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_cpu(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, cpu_usages: &[f32]) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" cpu ")
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // How many cores we can display given the inner height
    let max_rows = inner.height as usize;
    let bar_width: usize = 10;

    let lines: Vec<Line> = cpu_usages
        .iter()
        .enumerate()
        .take(max_rows)
        .map(|(i, &pct)| {
            let pct = pct.clamp(0.0, 100.0);
            let filled = ((pct / 100.0) * bar_width as f32).round() as usize;
            let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);

            let bar_color = if pct >= 80.0 {
                Color::Red
            } else if pct >= 50.0 {
                Color::Yellow
            } else {
                Color::Green
            };

            Line::from(vec![
                Span::styled(format!("{:>2} ", i), Style::default().fg(Color::DarkGray)),
                Span::styled(bar, Style::default().fg(bar_color)),
                Span::styled(format!(" {:>3.0}%", pct), Style::default().fg(Color::White)),
            ])
        })
        .collect();

    frame.render_widget(Paragraph::new(lines), inner);
}

fn row(label: &'static str, val: String, val_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{:<16}", label),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(val, val_style),
    ])
}

fn fmt_best(best: i32) -> String {
    if best < 0 {
        "— / 25".into()
    } else {
        format!("{best} / 25")
    }
}

fn fmt_rate(rate: f64) -> String {
    if rate <= 0.0 {
        return "—".into();
    }
    if rate >= 1e12 {
        format!("{:.1}T/s", rate / 1e12)
    } else if rate >= 1e9 {
        format!("{:.1}B/s", rate / 1e9)
    } else if rate >= 1e6 {
        format!("{:.1}M/s", rate / 1e6)
    } else if rate >= 1e3 {
        format!("{:.1}K/s", rate / 1e3)
    } else {
        format!("{rate:.0}/s")
    }
}

fn fmt_count(n: u64) -> String {
    if n == 0 {
        return "0".into();
    }
    if n >= 1_000_000_000_000 {
        format!("{:.2}T", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{n}")
    }
}

fn fmt_uptime(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{h}h {m}m {s}s")
}

fn best_char(best: i32) -> char {
    match best {
        n if n < 0 => '?',
        n => char::from_digit(n as u32, 36).unwrap_or('*'),
    }
}
