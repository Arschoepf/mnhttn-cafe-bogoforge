Bogo miner made in Rust

Has CPU, Nvidia CUDA and AMD HUP/ROCm miners

rust and c++:

`sudo dnf install git gcc gcc-c++ make pkgconf-pkg-config openssl-devel`

install rust:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

make config file
```cp conf.toml.example conf.toml```

Setting are in `conf.tml`:
```
[identity]
uuid = ""
nickname = ""
code = ""

[server]
url = "wss://bogo.swapjs.dev/ws"

[compute]
# cpu section
use_cpu = false
cpu_chunk_size = 50000000
cpu_threads = 0

# Nvidia backend
use_gpu = false
gpu_chunk_size = 536870912
cuda_blocks = 1024
cuda_threads_per_block = 256

# AMD/HIP backend
use_amd = false
amd_chunk_size = 535970912
amd_blocks = 2048
amd_threads_per_block = 256

[reporting]
report_interval = 1000
```


To Build AMD/HIP:
```
HIP_ARCH= ### GPU ARCHITECTURE ### \
ROCM_PATH=/usr \
RUSTFLAGS="-L native=/usr/lib64" \
cargo run --release --features hip
```

To run AMD/HIP:
```
ROCM_PATH=/usr \
RUSTFLAGS="-L native=/usr/lib64" \
HIP_ARCH=gfx1201 \
./target/release/bogoforge
```
or add flags to `.cargo/config.toml`:

