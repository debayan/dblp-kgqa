# Entity Linking

### Training:

```
python -u train.py --model_name t5-small --batch_size 4 --epochs 10 --lr 0.001
```
### Evaluating:

```
python -u evaluate.py --model_name t5-base --embedding_name transe
```
### Running Streamlit

```
streamlit run Home.py
```
