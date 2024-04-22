# intrinsic repo usage notes

## windows setup - fails?
- `git clone https://github.com/EarthenSky/IntrinsicCompositing`
- do `./cmpt461_env/Scripts/activate` before running anything
- `cd IntrinsicCompositing`
- `python3.11 -m pip install .`
- `cd interface`
  - ? `python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
  - ? `pip install gdown` 
  - ? install wget through windows
- `python3.11 interface.py --bg examples/bgs/lamp.jpeg --fg examples/fgs/soap.png --mask examples/masks/soap.png`

## wsl setup - see https://blog.safnet.com/archive/2022/01/21/python-from-wsl/
- sudo apt-get install python3-tk
- sudo apt-get install python3-venv
- (optional) sudo apt-get install python3-wheel
- python3 -m venv cmpt461_env
- source cmpt461_env/bin/activate
- cd IntrinsicCompositing
- python3 -m pip install .
- python3 -m pip install gdown
- windows...> $ choco install vcxsrv
  - setup as in the blog post. ie. run XLaunch
- update ~/.bashrc as in the linked stackoverflow
