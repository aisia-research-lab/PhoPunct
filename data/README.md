# Data

Please unzip two ZIP files to extract two folders: `News` and `Novels`. Place the corpus here (this directory's contents are git-ignored):

```
data/punctuation/
├── News/    train.txt  valid.txt  test.txt
└── Novels/  train.txt  valid.txt  test.txt
```

**Source:** TODO — cite / link where the News and Novels splits come from and their licence.

## Format

One `<word> <label>` pair per line, whitespace-separated; the whole file is one
continuous token stream (blank lines are ignored). Labels:
`O PERIOD COMMA COLON QMARK EXCLAM SEMICOLON` (a label marks the punctuation
that *follows* the word). Text is lower-cased and split into syllables (not word-segmented); links are replaced by the placeholder `link_obj`.

```
hôm O
nay O
trời O
đẹp O
quá COMMA
...
```

## Statistics (number of words and of each punctuation label)

| Domain | Split | Words | PERIOD | COMMA | COLON | QMARK | EXCLAM | SEMICOLON |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| News | train | 11,939,762 | 419,580 | 482,435 | 32,177 | 13,902 | 7,384 | 5,675 |
| News | valid | 4,003,912 | 140,170 | 162,056 | 10,825 | 4,884 | 2,556 | 1,944 |
| News | test | 3,958,621 | 138,967 | 160,472 | 10,728 | 4,468 | 2,333 | 2,045 |
| Novels | train | 1,350,500 | 66,519 | 50,909 | 742 | 14,899 | 30,183 | 48 |
| Novels | valid | 358,791 | 17,716 | 13,356 | 320 | 3,868 | 6,468 | 6 |
| Novels | test | 526,011 | 29,643 | 21,231 | 1,153 | 5,271 | 9,167 | 43 |

## Checksums (SHA-256) of the files used in the paper

```
beebe7921b3efbac0445adeb039785d392efb0ece5a4999b686c5099a66edbd3  News/train.txt
8dce3a222be3bd08b4315dd3499c7b2005e85d9b4ecf6a1afba8dded03e80527  News/valid.txt
89c1b9ad608fe192a9e1c1b88f05964a1a565b6a2fe5cd82972f6ffc9332ead8  News/test.txt
afae132917c3cebfd0dbcfe4bb2b6885660362a34bec9406eeaf1985d157a224  Novels/train.txt
67568c06380228c77c163497562b0fc4cc103f574f5a5a278fd38510767b7b60  Novels/valid.txt
42dc01f71ae9f6d5050926d389b6ed9fef61e38ce5ba2a01692f453bdd2c3df3  Novels/test.txt
```

Verify with `cd data/punctuation && sha256sum -c ../SHA256SUMS`.
