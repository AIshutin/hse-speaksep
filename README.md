# Speaker Separation

### How to run?

```
pip3 install -r requirements.txt
python3 create_mixture.py 0 # to create dataset
python3 train.py -c speaksep/configs/spexplus.json
```

### How to test quality?

```
gdown "https://drive.google.com/file/d/1J38XxOZklo4B4DRfpr9FajFfg9CxmSX7/view?usp=sharing" -O default_test_model/checkpoint.pth --fuzzy
python3 test.py -b 32 -t <your_data_folder>
```

Check HW description [here](https://github.com/XuMuK1/dla2023/tree/2023/hw2_ss).