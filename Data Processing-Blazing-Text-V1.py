{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "download: s3://amazon-reviews-pds/tsv/amazon_reviews_us_Camera_v1_00.tsv.gz to ../tmp/amazon_reviews_us_Camera_v1_00.tsv.gz\n"
     ]
    }
   ],
   "source": [
    "%%sh\n",
    "aws s3 cp s3://amazon-reviews-pds/tsv/amazon_reviews_us_Camera_v1_00.tsv.gz /tmp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
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
    "import pandas as pd\n",
    "data = pd.read_csv(\n",
    "    '/tmp/amazon_reviews_us_Camera_v1_00.tsv.gz',\n",
    "    sep='\\t', compression='gzip',\n",
    "    error_bad_lines=False, dtype='str')\n",
    "\n",
    "data.dropna(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1800755, 15)\n",
      "Index(['marketplace', 'customer_id', 'review_id', 'product_id',\n",
      "       'product_parent', 'product_title', 'product_category', 'star_rating',\n",
      "       'helpful_votes', 'total_votes', 'vine', 'verified_purchase',\n",
      "       'review_headline', 'review_body', 'review_date'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(data.shape)\n",
    "print(data.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data[:100000]\n",
    "data = data[['star_rating', 'review_body']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['label'] = data.star_rating.map({\n",
    "'1': '__label__negative__',\n",
    "'2': '__label__negative__',\n",
    "'3': '__label__neutral__',\n",
    "'4': '__label__positive__',\n",
    "'5': '__label__positive__'})\n",
    "data = data.drop(['star_rating'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review_body</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ok</td>\n",
       "      <td>__label__positive__</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Perfect, even sturdier than the original!</td>\n",
       "      <td>__label__positive__</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>If the words, &amp;#34;Cheap Chinese Junk&amp;#34; com...</td>\n",
       "      <td>__label__negative__</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Exactly what I wanted and expected. Perfect fo...</td>\n",
       "      <td>__label__positive__</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>I will look past the fact that they tricked me...</td>\n",
       "      <td>__label__positive__</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>100011</th>\n",
       "      <td>Wonderful product and well made.</td>\n",
       "      <td>__label__positive__</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>100012</th>\n",
       "      <td>Very pleased with the quality of the unit for ...</td>\n",
       "      <td>__label__positive__</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>100013</th>\n",
       "      <td>What can I say? I prefer Hoya.  I always buy H...</td>\n",
       "      <td>__label__positive__</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>100014</th>\n",
       "      <td>The battery works great.</td>\n",
       "      <td>__label__positive__</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>100015</th>\n",
       "      <td>The price is great. They fold up very small. P...</td>\n",
       "      <td>__label__positive__</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>100000 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review_body                label\n",
       "0                                                      ok  __label__positive__\n",
       "1               Perfect, even sturdier than the original!  __label__positive__\n",
       "2       If the words, &#34;Cheap Chinese Junk&#34; com...  __label__negative__\n",
       "3       Exactly what I wanted and expected. Perfect fo...  __label__positive__\n",
       "4       I will look past the fact that they tricked me...  __label__positive__\n",
       "...                                                   ...                  ...\n",
       "100011                   Wonderful product and well made.  __label__positive__\n",
       "100012  Very pleased with the quality of the unit for ...  __label__positive__\n",
       "100013  What can I say? I prefer Hoya.  I always buy H...  __label__positive__\n",
       "100014                           The battery works great.  __label__positive__\n",
       "100015  The price is great. They fold up very small. P...  __label__positive__\n",
       "\n",
       "[100000 rows x 2 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data[['label', 'review_body']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip -q install nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('punkt')\n",
    "data['review_body'] = data['review_body'].apply(nltk.\n",
    "word_tokenize)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['review_body'] = data.apply(lambda row: \" \".join(row['review_body']).lower(), axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "training, validation = train_test_split(data, test_size=0.05)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
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
