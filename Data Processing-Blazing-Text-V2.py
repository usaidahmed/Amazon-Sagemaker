{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import argparse, os, subprocess, sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "def install(package):\n",
    "    subprocess.call([sys.executable, \"-m\", \"pip\",\"install\", package])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__=='__main__':\n",
    "    install('nltk')\n",
    "    import nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "_StoreAction(option_strings=['--split-ratio'], dest='split_ratio', nargs=None, const=None, default=0.1, type=<class 'float'>, choices=None, help=None, metavar=None)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "parser = argparse.ArgumentParser()\n",
    "parser.add_argument('--filename', type=str)\n",
    "parser.add_argument('--num-reviews', type=int)\n",
    "parser.add_argument('--split-ratio', type=float,default=0.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "args, _ = parser.parse_known_args()\n",
    "filename = args.filename\n",
    "num_reviews = args.num_reviews\n",
    "split_ratio = args.split_ratio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "input_data_path = os.path.join('/tmp/amazon_reviews_us_Camera_v1_00.tsv.gz')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "b'Skipping line 85458: expected 15 fields, saw 22\\nSkipping line 91161: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 166123: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 225458: expected 15 fields, saw 22\\nSkipping line 229936: expected 15 fields, saw 22\\nSkipping line 259297: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 284728: expected 15 fields, saw 22\\nSkipping line 286334: expected 15 fields, saw 22\\nSkipping line 293400: expected 15 fields, saw 22\\nSkipping line 294415: expected 15 fields, saw 22\\nSkipping line 308150: expected 15 fields, saw 22\\nSkipping line 315022: expected 15 fields, saw 22\\nSkipping line 315730: expected 15 fields, saw 22\\nSkipping line 316071: expected 15 fields, saw 22\\nSkipping line 326729: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 329101: expected 15 fields, saw 22\\nSkipping line 333077: expected 15 fields, saw 22\\nSkipping line 377031: expected 15 fields, saw 22\\nSkipping line 389496: expected 15 fields, saw 22\\nSkipping line 390486: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 418308: expected 15 fields, saw 22\\nSkipping line 454332: expected 15 fields, saw 22\\nSkipping line 458342: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 460704: expected 15 fields, saw 22\\nSkipping line 466250: expected 15 fields, saw 22\\nSkipping line 486023: expected 15 fields, saw 22\\nSkipping line 492819: expected 15 fields, saw 22\\nSkipping line 517468: expected 15 fields, saw 22\\nSkipping line 520963: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 528810: expected 15 fields, saw 22\\nSkipping line 554419: expected 15 fields, saw 22\\nSkipping line 565266: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 613248: expected 15 fields, saw 22\\nSkipping line 613988: expected 15 fields, saw 22\\nSkipping line 620134: expected 15 fields, saw 22\\nSkipping line 642170: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 670152: expected 15 fields, saw 22\\nSkipping line 681751: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 811638: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 913254: expected 15 fields, saw 22\\n'\n",
      "b'Skipping line 1168305: expected 15 fields, saw 22\\n'\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv(input_data_path, sep='\\t',\n",
    "                   compression='gzip', error_bad_lines=False,\n",
    "                   dtype='str')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "training, validation = train_test_split(data, test_size=split_ratio)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.savetxt('opt/ml/processing/train/training.txt', training.values,fmt='%s')\n",
    "np.savetxt('opt/ml/processing/validation/validation.txt', validation.values,fmt='%s')"
   ]
  }
 ],
 "metadata": {
  "instance_type": "ml.g4dn.xlarge",
  "kernelspec": {
   "display_name": "Python 3 (TensorFlow 2.3 Python 3.7 GPU Optimized)",
   "language": "python",
   "name": "python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-2:429704687514:image/tensorflow-2.3-gpu-py37-cu110-ubuntu18.04-v3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
