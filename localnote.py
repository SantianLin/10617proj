from datasets import load_dataset
import collections

raw_datasets = load_dataset("midas/inspec")
print(raw_datasets)

# sentence length, stat of labels, first word label,

# train set stats
tag_count = 0
length = 0
counter = collections.Counter()
for data in raw_datasets["train"]:
    counter0 = collections.Counter(data['doc_bio_tags'])
    # print(len(data['doc_bio_tags']))
    length += len(data['doc_bio_tags'])/1000
    counter += counter0
    # print(counter)
print(counter)
print(length)
total = sum(counter.values())
for key in counter:
    counter[key] /= total
print(counter)

# test set stats
tag_count = 0
length = 0
counter = collections.Counter()
for data in raw_datasets["test"]:
    counter0 = collections.Counter(data['doc_bio_tags'])
    length += len(data['doc_bio_tags'])/500
    counter += counter0
    # print(counter)
print(counter)
print(length)
total = sum(counter.values())
for key in counter:
    counter[key] /= total
print(counter)
