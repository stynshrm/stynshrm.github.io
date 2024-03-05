---
author: Satyan Sharma
title: Abseil flags
date: 2022-05-31
math: true
tags: ["Python"]
thumbnail: /th-abseil.png
---

## Abseil Python Library

While working on one of python projects, I encountered abseil python library. I found 
out `absl.flags` quiet interesting. It provides easy way for argument processing 
and replaces `getopt()` and `optparse`. One is able to provide default values and also
auto-generate help documents. 

```python
from absl import flags
from absl import app

# Flag names are globally defined!
flags.DEFINE_integer('job_id', 32, 'Job ID.', lower_bound=0)
flags.DEFINE_string('data_dir', None, 'Path to data directory.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_enum('job_name', 'production',
                  ['production', 'test'],
                  'Choose  job configuration - '
                  'smaller test job database (test) '
                  'or full production run (production)')

flags.mark_flags_as_required([
    'output_dir',
    'data_dir'])

FLAGS = flags.FLAGS


def main(argv):
    
    num_nodes = 8 if FLAGS.job_name == "test" else 32
    print(f"You are runing a {FLAGS.job_name} job with {num_nodes} nodes")
    print(FLAGS.data_dir)
    print(FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)

```

Runing without arguments:

```
python test.py

FATAL Flags parsing error:
  flag --output_dir=None: Flag --output_dir must have a value other than None.
  flag --data_dir=None: Flag --data_dir must have a value other than None.
Pass --helpshort or --helpfull to see help on flags.
```

To see help documentation
```
python test.py --help

  --data_dir: Path to data directory.
  --job_id: Job ID.
    (default: '32')
    (a non-negative integer)
  --job_name: <production|test>: Choose  job configuration - smaller test job database (test) or full production run (production)
    (default: 'production')
  --output_dir: Path to a directory that will store the results.
```

And with the required arguments

```
python test.py --output_dir="/home/user/out_dir" --data_dir="/home/user/data_dir/"

You are runing a production job with 32 nodes
/home/user/data_dir/
/home/user/out_dir
```
