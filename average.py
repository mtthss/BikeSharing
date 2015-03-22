from __future__ import division, print_function

import pandas as pd
import numpy as np
import os

from util.const import *
from util.util import *

folder = PREDICTION_FILE[3:]
p = {
    'average': True,
    'hashes': ['-1603307410989466001', '5986558112425607447']
}


count = np.zeros((N_TEST_SAMPLES,))
for h in p['hashes']:
    df = pd.read_csv(folder % h)
    count += df['count']

count /= len(p['hashes'])

os.chdir('./model')
generate_submission(df, count, p)

