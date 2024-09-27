import torch
from params import p

def getI(c):
    for i in range(p.len):
        if c == p.symbols[i]:
            return i
    return None

def textToSeq(text):
  text = text.lower()
  seq = []
  for c in text:
    i = getI(c)
    if i is not None:
      seq.append(i)

  seq.append(getI("EOS"))

  return torch.IntTensor(seq);

if __name__ == "__main__":
    print(textToSeq("Test test test"))