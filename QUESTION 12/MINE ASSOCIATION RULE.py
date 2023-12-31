# -*- coding: utf-8 -*-
"""

@author: Hp
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

new_dataset = [['Bread', 'Milk', 'Eggs'],
               ['Cheese', 'Onion', 'Eggs', 'Yogurt'],
               ['Milk', 'Onion', 'Eggs', 'Yogurt'],
               ['Bread', 'Cheese', 'Eggs'],
               ['Bread', 'Milk', 'Yogurt']]

te_new = TransactionEncoder()
te_ary_new = te_new.fit(new_dataset).transform(new_dataset)
df_new = pd.DataFrame(te_ary_new, columns=te_new.columns_)

frequent_itemsets_new = apriori(df_new, min_support=0.4, use_colnames=True)

rules_new = association_rules(frequent_itemsets_new, metric="confidence", min_threshold=0.6)

print(rules_new)