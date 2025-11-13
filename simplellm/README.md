## SimpleLLM for Qwen 0.6B

### install
```bash
git clone https://github.com/gbdjxgp/SimpleLLM.git
cd SimpleLLM
pip install uv
# or install torch with cuda
uv pip install torch_npu
uv sync
# then download the pretrained model use modelscope/huggingface
modelscope download xxx
# next, install simplellm package or run your python code in this directory.
# install simplellm in develop mode
pip install -v -e .
# run code in this directory
python test_qwen3_0_6b.py
