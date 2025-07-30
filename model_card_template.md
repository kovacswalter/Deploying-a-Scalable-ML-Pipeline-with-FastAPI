# Model Card
Mason Lyman
Udemy Deploying a Scalable ML Pipeline with FastAPI

## Model Details
This model takes U.S. census data and builds a RandomForestClassifier model to predict salary of an individual based on that data.
## Intended Use
This model is intended to predict salary of an individual based on census data. Specifically whether an individual makes more or less than $50,000 a year.
## Training Data
I split the data on an 80:20 split, using 80% for training data.
## Evaluation Data
The evaluation data was the remaining 20%, from the 80:20 split.
## Metrics
We used Percision, Recall, and F1 as our metrics for model performance. Percision shows us the percentage of true positives out of all the positives the model returned. Recall shows the percentage of positves out of all true positives and fals negatives. F1 is an average between the two.
### Overall Metrics
Precision: 0.7255 | Recall: 0.6151 | F1: 0.6658

### Workclass Metrics
workclass: ?, Count: 362
Precision: 0.6061 | Recall: 0.4878 | F1: 0.5405
workclass: Federal-gov, Count: 191
Precision: 0.7692 | Recall: 0.6667 | F1: 0.7143
workclass: Local-gov, Count: 379
Precision: 0.6321 | Recall: 0.5583 | F1: 0.5929
workclass: Never-worked, Count: 4
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
workclass: Private, Count: 4,572
Precision: 0.7206 | Recall: 0.6099 | F1: 0.6606
workclass: Self-emp-inc, Count: 230
Precision: 0.8029 | Recall: 0.8800 | F1: 0.8397
workclass: Self-emp-not-inc, Count: 509
Precision: 0.7300 | Recall: 0.4803 | F1: 0.5794
workclass: State-gov, Count: 263
Precision: 0.7966 | Recall: 0.6267 | F1: 0.7015
workclass: Without-pay, Count: 3
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
### Education Metrics
education: 10th, Count: 174
Precision: 0.7778 | Recall: 0.4118 | F1: 0.5385
education: 11th, Count: 216
Precision: 0.7500 | Recall: 0.3333 | F1: 0.4615
education: 12th, Count: 105
Precision: 0.6000 | Recall: 0.3750 | F1: 0.4615
education: 1st-4th, Count: 31
Precision: 1.0000 | Recall: 0.0000 | F1: 0.0000
education: 5th-6th, Count: 62
Precision: 1.0000 | Recall: 0.5000 | F1: 0.6667
education: 7th-8th, Count: 126
Precision: 1.0000 | Recall: 0.0909 | F1: 0.1667
education: 9th, Count: 96
Precision: 1.0000 | Recall: 0.2500 | F1: 0.4000
education: Assoc-acdm, Count: 211
Precision: 0.6226 | Recall: 0.5893 | F1: 0.6055
education: Assoc-voc, Count: 263
Precision: 0.6863 | Recall: 0.5833 | F1: 0.6306
education: Bachelors, Count: 1,072
Precision: 0.7768 | Recall: 0.7160 | F1: 0.7452
education: Doctorate, Count: 96
Precision: 0.8219 | Recall: 0.8824 | F1: 0.8511
education: HS-grad, Count: 2,146
Precision: 0.5638 | Recall: 0.3881 | F1: 0.4597
education: Masters, Count: 329
Precision: 0.8324 | Recall: 0.8063 | F1: 0.8191
education: Preschool, Count: 13
Precision: 0.0000 | Recall: 1.0000 | F1: 0.0000
education: Prof-school, Count: 112
Precision: 0.8941 | Recall: 0.9048 | F1: 0.8994
education: Some-college, Count: 1,461
Precision: 0.6364 | Recall: 0.5081 | F1: 0.5650
### Marital Status Metrics
marital-status: Divorced, Count: 866
Precision: 0.7949 | Recall: 0.3780 | F1: 0.5124
marital-status: Married-AF-spouse, Count: 4
Precision: 1.0000 | Recall: 0.0000 | F1: 0.0000
marital-status: Married-civ-spouse, Count: 3,030
Precision: 0.7125 | Recall: 0.6644 | F1: 0.6876
marital-status: Married-spouse-absent, Count: 88
Precision: 0.7500 | Recall: 0.4286 | F1: 0.5455
marital-status: Never-married, Count: 2,102
Precision: 0.9535 | Recall: 0.3475 | F1: 0.5093
marital-status: Separated, Count: 223
Precision: 1.0000 | Recall: 0.3478 | F1: 0.5161
marital-status: Widowed, Count: 200
Precision: 1.0000 | Recall: 0.2857 | F1: 0.4444
### Occupation Metrics
occupation: ?, Count: 366
Precision: 0.6061 | Recall: 0.4878 | F1: 0.5405
occupation: Adm-clerical, Count: 767
Precision: 0.7042 | Recall: 0.4132 | F1: 0.5208
occupation: Armed-Forces, Count: 1
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
occupation: Craft-repair, Count: 869
Precision: 0.5755 | Recall: 0.4420 | F1: 0.5000
occupation: Exec-managerial, Count: 799
Precision: 0.7995 | Recall: 0.7549 | F1: 0.7765
occupation: Farming-fishing, Count: 202
Precision: 0.5833 | Recall: 0.3043 | F1: 0.4000
occupation: Handlers-cleaners, Count: 288
Precision: 0.4444 | Recall: 0.2000 | F1: 0.2759
occupation: Machine-op-inspct, Count: 382
Precision: 0.4722 | Recall: 0.3542 | F1: 0.4048
occupation: Other-service, Count: 648
Precision: 0.6250 | Recall: 0.1562 | F1: 0.2500
occupation: Priv-house-serv, Count: 33
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
occupation: Prof-specialty, Count: 826
Precision: 0.7704 | Recall: 0.7624 | F1: 0.7664
occupation: Protective-serv, Count: 120
Precision: 0.6765 | Recall: 0.5476 | F1: 0.6053
occupation: Sales, Count: 683
Precision: 0.7105 | Recall: 0.6391 | F1: 0.6729
occupation: Tech-support, Count: 208
Precision: 0.7273 | Recall: 0.6557 | F1: 0.6897
occupation: Transport-moving, Count: 321
Precision: 0.6829 | Recall: 0.4058 | F1: 0.5091
### Relationship Metrics
relationship: Husband, Count: 2,659
Precision: 0.7129 | Recall: 0.6686 | F1: 0.6900
relationship: Not-in-family, Count: 1,685
Precision: 0.9012 | Recall: 0.3822 | F1: 0.5368
relationship: Other-relative, Count: 177
Precision: 0.8571 | Recall: 0.5000 | F1: 0.6316
relationship: Own-child, Count: 990
Precision: 1.0000 | Recall: 0.2500 | F1: 0.4000
relationship: Unmarried, Count: 683
Precision: 0.7692 | Recall: 0.2273 | F1: 0.3509
relationship: Wife, Count: 319
Precision: 0.7080 | Recall: 0.6382 | F1: 0.6713
### Racial Identity Metrics
race: Amer-Indian-Eskimo, Count: 57
Precision: 0.7500 | Recall: 0.6000 | F1: 0.6667
race: Asian-Pac-Islander, Count: 209
Precision: 0.6500 | Recall: 0.4643 | F1: 0.5417
race: Black, Count: 600
Precision: 0.7778 | Recall: 0.5185 | F1: 0.6222
race: Other, Count: 54
Precision: 1.0000 | Recall: 0.6000 | F1: 0.7500
race: White, Count: 5,593
Precision: 0.7249 | Recall: 0.6265 | F1: 0.6721
### Gender Metrics
sex: Female, Count: 2,129
Precision: 0.7529 | Recall: 0.5198 | F1: 0.6150
sex: Male, Count: 4,384
Precision: 0.7215 | Recall: 0.6329 | F1: 0.6743
### Country of Origin Metrics
native-country: ?, Count: 125
Precision: 0.6250 | Recall: 0.4688 | F1: 0.5357
native-country: Cambodia, Count: 3
Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000
native-country: Canada, Count: 23
Precision: 0.7778 | Recall: 0.7000 | F1: 0.7368
native-country: China, Count: 10
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: Columbia, Count: 13
Precision: 0.0000 | Recall: 1.0000 | F1: 0.0000
native-country: Cuba, Count: 19
Precision: 0.7500 | Recall: 0.5000 | F1: 0.6000
native-country: Dominican-Republic, Count: 8
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: Ecuador, Count: 6
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: El-Salvador, Count: 17
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: England, Count: 18
Precision: 0.8000 | Recall: 0.5000 | F1: 0.6154
native-country: France, Count: 5
Precision: 0.6667 | Recall: 1.0000 | F1: 0.8000
native-country: Germany, Count: 32
Precision: 0.7778 | Recall: 0.7000 | F1: 0.7368
native-country: Greece, Count: 7
Precision: 0.5000 | Recall: 1.0000 | F1: 0.6667
native-country: Guatemala, Count: 12
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: Haiti, Count: 6
Precision: 1.0000 | Recall: 0.0000 | F1: 0.0000
native-country: Holand-Netherlands, Count: 1
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: Honduras, Count: 5
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: Hong, Count: 2
Precision: 1.0000 | Recall: 0.5000 | F1: 0.6667
native-country: Hungary, Count: 2
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: India, Count: 21
Precision: 0.3333 | Recall: 0.5000 | F1: 0.4000
native-country: Iran, Count: 8
Precision: 0.0000 | Recall: 1.0000 | F1: 0.0000
native-country: Ireland, Count: 6
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: Italy, Count: 9
Precision: 0.5000 | Recall: 1.0000 | F1: 0.6667
native-country: Jamaica, Count: 17
Precision: 1.0000 | Recall: 0.0000 | F1: 0.0000
native-country: Japan, Count: 12
Precision: 0.5000 | Recall: 0.6667 | F1: 0.5714
native-country: Laos, Count: 3
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: Mexico, Count: 132
Precision: 0.5000 | Recall: 0.1429 | F1: 0.2222
native-country: Nicaragua, Count: 8
Precision: 1.0000 | Recall: 0.0000 | F1: 0.0000
native-country: Outlying-US(Guam-USVI-etc), Count: 1
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: Peru, Count: 9
Precision: 1.0000 | Recall: 0.5000 | F1: 0.6667
native-country: Philippines, Count: 39
Precision: 0.8182 | Recall: 0.6000 | F1: 0.6923
native-country: Poland, Count: 14
Precision: 0.6667 | Recall: 0.6667 | F1: 0.6667
native-country: Portugal, Count: 4
Precision: 1.0000 | Recall: 0.0000 | F1: 0.0000
native-country: Puerto-Rico, Count: 28
Precision: 0.6000 | Recall: 1.0000 | F1: 0.7500
native-country: Scotland, Count: 3
Precision: 1.0000 | Recall: 0.5000 | F1: 0.6667
native-country: South, Count: 18
Precision: 1.0000 | Recall: 0.3333 | F1: 0.5000
native-country: Taiwan, Count: 9
Precision: 1.0000 | Recall: 0.7500 | F1: 0.8571
native-country: Thailand, Count: 5
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: Trinadad&Tobago, Count: 2
Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
native-country: United-States, Count: 5,832
Precision: 0.7312 | Recall: 0.6218 | F1: 0.6721
native-country: Vietnam, Count: 14
Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000
native-country: Yugoslavia, Count: 5
Precision: 0.5000 | Recall: 0.6667 | F1: 0.5714
## Ethical Considerations
Since this model deals with demographic data it may represent historical bias in that are present in the data. An example is that looking at percision metrics for American Indians we see a drop when compared to the others.
## Caveats and Recommendations
There is limited data when it comes to some of the differnet groups within categories. For example we have several groups in the native-country where there are five or less data points. To remedy this my recomendation is to increase the size of the data set.