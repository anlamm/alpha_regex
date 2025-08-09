# Alpha_regex

This is a python implementation of alpha_regex synthesizer that output regex from positive and negative examples. 
The algorithm is from the paper ```Synthesizing Regular Expressions from Examples for Introductory Automata Assignments``` https://dl.acm.org/doi/pdf/10.1145/3093335.2993244

### Use

To use testcases in ```testcase``` folder

```python test.py```

To input examples manually

```python main.py```

Then,

- Enter positive strings after ```Enter valid strings (e.g. ["123", "456"]) : ``` in the same formate as the examples
- Enter negative strings after ```Enter invalid strings (e.g. ["abc", "def"]) : ``` in the same formate as the examples
- Output will be generated after ```Regex generated: ```