# Evalutation of Large Language Model

## Data

**Measuring Massive Multitask Language Understanding**

- 57가지 영역에서 전반적인 LLM 성능을 측정할 수 있는 데이터셋

**wikitext**

- ppl 측정을 위한 데이터셋

## Prompt

### Few-shot (5)
```
The following are multiple choice questions (with answers) about  abstract `subject`.

Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
A. 0
B. 1
C. 2
D. 3
Answer: B

Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.
A. True, True
B. False, False
C. True, False
D. False, True
Answer: B

Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.
A. True, True
B. False, False
C. True, False
D. False, True
Answer: C

Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.
A. True, True
B. False, False
C. True, False
D. False, True
Answer: A

Find the characteristic of the ring 2Z.
A. 0
B. 3
C. 12
D. 30
Answer: A

Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
A. 0
B. 4
C. 2
D. 6
Answer:
```

### Zero-shot

```
The following are multiple choice questions (with answers) about  abstract `subject`.

Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
A. 0
B. 4
C. 2
D. 6
Answer:
```

## Config

필요에 따라 `config.json`파일을 수정해서 사용.

- dev_dir : few-shot 디렉토리

- test_dir : 성능 측정 디렉토리

- model_name : LLM 모델명

- sub_categories : 57가지 subject를 17개의 서브 카테고리로 분류

- categories : 크게 `STEM`, `humanities`, `social sciences`, `other`로 분류

- few_shot : few-shot 프롬프트를 사용할지 여부(True/False)

- ppl : perplexity 측정할지 여부(True/False)

- stride : ppl 측정시 사용되는 stride

## Usage & Output

```
>> python main.py
```
_output_
```
================= Start evaluation =================
Subject                                      Acc  
--------------------------------------------------
abstract_algebra              Text match     0.260
                              Label match    0.290
                              Last logits    0.000
...

```

## TODO

- Last logits 관련 문제 해결

- ppl의 경우 llm 마다 `max_length`가져오는 법이 다름.
