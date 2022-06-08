# Vocabulary lists
The corpus is generated from templates where the vocabulary is selected from the following vocabulary lists, containing words annotated for their tense and role in the templates ensuring grammatical sentences.

The files in this folder are as follows:
1. verblist_T_usf_freq.csv
2. verblist_DT_usf_freq.csv
3. nounlist_usf_freq.csv
4. adjective_voc_list_USF.csv
5. complexNPS.csv


### 1. Verb Transitive
This file contains 48 transitive verbs. These are annotated with which categories of nouns they can be used with.

### 2. Verb Di-Transitive
This file contains 22 ditransitive verbs. These are annotated with which categories of nouns they can be used with.

### 3. Noun
This file contains 119 Nouns labeled by their type and role in a template.

Noun categories to fill in the N-slots of verbs in
verblist_T_usf_freq.csv and verblist_DT_usef_freq.csv

"category" and "cat_main" refer to columns in nounlist_usf_freq.csv


- Person: category=person
- Group: category=group
- Object: category=object
- Drinkable = category=drinkable
- Edible = category=edible

- Human: cat_main=human
- OED: cat_main=object

- PO: category=person OR category=object
- POED: category=person OR category=object OR category=edible OR category=drinkable

### 4. Adjective
This file contains 155 Adjectives annotated with which Noun is suitable for them to modify.

### 5. Complex NPs
This file contains 50 additional noun phrase modifier templates which can be added to nouns in the template sentences.

### 6. Intransitive
6 intransitive verbs were chosen to be used as intervening material in the corpora designed to test recency. They are: remain, exist, stay, come, go, appear


# Templates

### Transitive

**Active:** 'the < n1 > < v > the < n2 > .'

**Passive:** 'the < n2 > was < v > by the < n1 > .'

### Dative

**Prepositional Object:** 'the < n1 > < v > the < n2 > < prep  > the < n3 > .'

**Double Object:** 'the < n1 > < v > the < n3 > the < n2 >.'

## Notes
- < prep > is looked up for each verb from list of prepositions designed to result in both the po and do template carrying the same semantic meaning for any shared verb.

- In all cases for ditransitive templates, < n1 > and < n3 > were chosen from the human list of nouns, and < n2 > from the non-human.

### Vocabulary selection notes
Transitive and Ditransitive verblists are in this case distinct from one anther. in the case where a verb is both transitive and ditranstive, the ditransitive label is chosen due to this being the rarer verb type.

Simlex and Simverb were merged, and our selected vocabulary is the set of all verbs labeled for either transitive or ditransitive. This labelling was performed using the below verb-lists. For each verb, its frequency is labeled using cocoa values.

**Transitive** labels come from verb lists merged from the following sources:
https://englishpost.org/transitive-verbs-list/#Transitive_Verbs_List_A_to_F


**Ditransitive** labels come from the following verb lists merged:
https://www.cse.unsw.edu.au/~billw/ditransitive.html
di-transitive: http://www.aprendeinglesenleganes.com/resources/

**intranstitive** six verbs come from Ivanova et al. ('remain', 'exist', 'dwell', 'stay', 'peek', 'doze') however not all are in the same frequent category as in our corpus design: thus we select only the three that meet this criteria, adding ('come', 'go', 'appear') to the list in order that we have sufficient sample size. These three appear in simlex/simverb and are frequent enough for our corpus creation criteria.
