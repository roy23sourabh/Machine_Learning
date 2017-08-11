# Week 4 assigment;Weight lifting dataset 
Sourabh Roy  
8/11/2017  



## Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Downloading data

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile = "./train.csv" )
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile = "./test.csv" )
```
## Loading Data

```r
training <- data.table::fread("./train.csv",sep=",",header = TRUE,stringsAsFactors = FALSE,na.strings = c("NA",""))

testing <- data.table::fread("./test.csv",sep = ",",header = TRUE,stringsAsFactors = FALSE,na.strings = c("NA",""))
```

## Tidying  Data  

Check  data for  missing values ,if missing values are way too much we will ignore the variable(easiest way),if missing values are within permissible limit we can go for imputing missing data as a pre process before we start modelling.

```r
dim(training)
sapply(training, function(x) sum(is.na(x)))
sapply(testing, function(x) sum(is.na(x)))
```
We can observe that we have 19622 observations 160 variables in the training data set 
out of which few variables have 19216 missing values(or 97.93%) which is way too much to include them in our model.Hence we have to give these variables a skip.


```r
Var <- sapply(training, function(x) ifelse(sum(is.na(x)) >0 ,"Exclude","Include"))
#We have variables that needs to be excluded
exVar <- which(Var == "Exclude")
```
New dataset for training and testing data.Also we observe that the variable such as user_name and V1, raw_timestamps and cvtd_timestamp is not useful for our analysis.
Hence our final training and testing data set as below

```r
trainDF <- dplyr::select(training,-c(exVar,1:5))
testDF <- dplyr::select(testing,-c(exVar,1:5))
```

## Exploratory Data Analysis

We are interested to identify how "classe" is related with other variables such as accelerometers on the belt, forearm, arm, and dumbell.Also correlation with other variables.
Based on  indvidual plots from training data we can come to conclusion that we have a non linear data.Hence linear regression models would not be a choice.We would start with  decision tree/classification to predict the outcome and further with other models as and when required for accuracy.



```r
set.seed(1245)
##Classification Tree
modFit <- train(classe~.,method="rpart",data = trainDF,trControl=trainControl(allowParallel = TRUE))
pred1 <- predict(modFit,trainDF)
confusionMatrix(pred1,trainDF$classe)  ## 49.56% accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5080 1581 1587 1449  524
##          B   81 1286  108  568  486
##          C  405  930 1727 1199  966
##          D    0    0    0    0    0
##          E   14    0    0    0 1631
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4956          
##                  95% CI : (0.4885, 0.5026)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3407          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9104  0.33869  0.50468   0.0000  0.45218
## Specificity            0.6339  0.92145  0.78395   1.0000  0.99913
## Pos Pred Value         0.4970  0.50850  0.33040      NaN  0.99149
## Neg Pred Value         0.9468  0.85310  0.88225   0.8361  0.89008
## Prevalence             0.2844  0.19351  0.17440   0.1639  0.18382
## Detection Rate         0.2589  0.06554  0.08801   0.0000  0.08312
## Detection Prevalence   0.5209  0.12889  0.26638   0.0000  0.08383
## Balanced Accuracy      0.7721  0.63007  0.64431   0.5000  0.72565
```

```r
## Linear Discriminant Analyses
modFit2 <- train(classe~.,data = trainDF,method = "lda")
pred2 <- predict(modFit2,trainDF)
confusionMatrix(pred2,trainDF$classe) ## 71.63%  
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4615  520  332  173  144
##          B  164 2493  320  132  528
##          C  366  473 2271  395  321
##          D  415  154  401 2401  338
##          E   20  157   98  115 2276
## 
## Overall Statistics
##                                         
##                Accuracy : 0.7163        
##                  95% CI : (0.71, 0.7226)
##     No Information Rate : 0.2844        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.6411        
##  Mcnemar's Test P-Value : < 2.2e-16     
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8271   0.6566   0.6636   0.7466   0.6310
## Specificity            0.9167   0.9277   0.9040   0.9203   0.9756
## Pos Pred Value         0.7979   0.6855   0.5936   0.6473   0.8537
## Neg Pred Value         0.9303   0.9184   0.9271   0.9488   0.9215
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2352   0.1271   0.1157   0.1224   0.1160
## Detection Prevalence   0.2948   0.1854   0.1950   0.1890   0.1359
## Balanced Accuracy      0.8719   0.7921   0.7838   0.8334   0.8033
```

```r
## Boosting with cross validation so as to avoid overfitting ( with K=4 folds)
modFit4 <- train(classe~.,method="gbm",data = trainDF,trControl=trainControl(allowParallel = TRUE,method = "cv",number = 4))
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1267
##      2        1.5243             nan     0.1000    0.0858
##      3        1.4671             nan     0.1000    0.0667
##      4        1.4236             nan     0.1000    0.0540
##      5        1.3889             nan     0.1000    0.0443
##      6        1.3596             nan     0.1000    0.0455
##      7        1.3307             nan     0.1000    0.0365
##      8        1.3077             nan     0.1000    0.0365
##      9        1.2814             nan     0.1000    0.0380
##     10        1.2573             nan     0.1000    0.0315
##     20        1.0951             nan     0.1000    0.0187
##     40        0.9142             nan     0.1000    0.0088
##     60        0.7993             nan     0.1000    0.0076
##     80        0.7175             nan     0.1000    0.0047
##    100        0.6522             nan     0.1000    0.0036
##    120        0.5967             nan     0.1000    0.0017
##    140        0.5511             nan     0.1000    0.0028
##    150        0.5309             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1893
##      2        1.4854             nan     0.1000    0.1274
##      3        1.4015             nan     0.1000    0.1079
##      4        1.3321             nan     0.1000    0.0875
##      5        1.2766             nan     0.1000    0.0687
##      6        1.2319             nan     0.1000    0.0762
##      7        1.1844             nan     0.1000    0.0636
##      8        1.1441             nan     0.1000    0.0502
##      9        1.1113             nan     0.1000    0.0439
##     10        1.0822             nan     0.1000    0.0483
##     20        0.8480             nan     0.1000    0.0329
##     40        0.6133             nan     0.1000    0.0166
##     60        0.4716             nan     0.1000    0.0056
##     80        0.3913             nan     0.1000    0.0056
##    100        0.3211             nan     0.1000    0.0059
##    120        0.2694             nan     0.1000    0.0019
##    140        0.2307             nan     0.1000    0.0022
##    150        0.2161             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2393
##      2        1.4589             nan     0.1000    0.1596
##      3        1.3570             nan     0.1000    0.1250
##      4        1.2789             nan     0.1000    0.1056
##      5        1.2115             nan     0.1000    0.0958
##      6        1.1503             nan     0.1000    0.0734
##      7        1.1028             nan     0.1000    0.0664
##      8        1.0599             nan     0.1000    0.0747
##      9        1.0143             nan     0.1000    0.0627
##     10        0.9753             nan     0.1000    0.0555
##     20        0.7046             nan     0.1000    0.0247
##     40        0.4582             nan     0.1000    0.0097
##     60        0.3287             nan     0.1000    0.0069
##     80        0.2482             nan     0.1000    0.0065
##    100        0.1906             nan     0.1000    0.0027
##    120        0.1499             nan     0.1000    0.0025
##    140        0.1215             nan     0.1000    0.0024
##    150        0.1082             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1321
##      2        1.5239             nan     0.1000    0.0891
##      3        1.4651             nan     0.1000    0.0648
##      4        1.4211             nan     0.1000    0.0531
##      5        1.3856             nan     0.1000    0.0483
##      6        1.3533             nan     0.1000    0.0405
##      7        1.3278             nan     0.1000    0.0426
##      8        1.3017             nan     0.1000    0.0387
##      9        1.2777             nan     0.1000    0.0357
##     10        1.2538             nan     0.1000    0.0292
##     20        1.0958             nan     0.1000    0.0184
##     40        0.9149             nan     0.1000    0.0080
##     60        0.8047             nan     0.1000    0.0066
##     80        0.7210             nan     0.1000    0.0050
##    100        0.6537             nan     0.1000    0.0047
##    120        0.6009             nan     0.1000    0.0037
##    140        0.5552             nan     0.1000    0.0033
##    150        0.5348             nan     0.1000    0.0026
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1924
##      2        1.4865             nan     0.1000    0.1285
##      3        1.4039             nan     0.1000    0.1087
##      4        1.3341             nan     0.1000    0.0884
##      5        1.2784             nan     0.1000    0.0775
##      6        1.2296             nan     0.1000    0.0637
##      7        1.1891             nan     0.1000    0.0672
##      8        1.1474             nan     0.1000    0.0504
##      9        1.1156             nan     0.1000    0.0478
##     10        1.0852             nan     0.1000    0.0512
##     20        0.8533             nan     0.1000    0.0250
##     40        0.6285             nan     0.1000    0.0134
##     60        0.4910             nan     0.1000    0.0082
##     80        0.4030             nan     0.1000    0.0071
##    100        0.3356             nan     0.1000    0.0034
##    120        0.2802             nan     0.1000    0.0019
##    140        0.2358             nan     0.1000    0.0041
##    150        0.2184             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2439
##      2        1.4561             nan     0.1000    0.1600
##      3        1.3557             nan     0.1000    0.1318
##      4        1.2733             nan     0.1000    0.1117
##      5        1.2024             nan     0.1000    0.0935
##      6        1.1442             nan     0.1000    0.0776
##      7        1.0956             nan     0.1000    0.0742
##      8        1.0496             nan     0.1000    0.0688
##      9        1.0074             nan     0.1000    0.0524
##     10        0.9737             nan     0.1000    0.0550
##     20        0.6991             nan     0.1000    0.0203
##     40        0.4570             nan     0.1000    0.0106
##     60        0.3298             nan     0.1000    0.0082
##     80        0.2542             nan     0.1000    0.0055
##    100        0.1985             nan     0.1000    0.0019
##    120        0.1597             nan     0.1000    0.0035
##    140        0.1267             nan     0.1000    0.0023
##    150        0.1139             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1288
##      2        1.5232             nan     0.1000    0.0894
##      3        1.4642             nan     0.1000    0.0667
##      4        1.4200             nan     0.1000    0.0536
##      5        1.3845             nan     0.1000    0.0527
##      6        1.3511             nan     0.1000    0.0424
##      7        1.3232             nan     0.1000    0.0347
##      8        1.3001             nan     0.1000    0.0379
##      9        1.2746             nan     0.1000    0.0335
##     10        1.2526             nan     0.1000    0.0301
##     20        1.0921             nan     0.1000    0.0193
##     40        0.9104             nan     0.1000    0.0089
##     60        0.7996             nan     0.1000    0.0064
##     80        0.7182             nan     0.1000    0.0049
##    100        0.6544             nan     0.1000    0.0050
##    120        0.5997             nan     0.1000    0.0043
##    140        0.5531             nan     0.1000    0.0026
##    150        0.5329             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1958
##      2        1.4844             nan     0.1000    0.1354
##      3        1.3984             nan     0.1000    0.1060
##      4        1.3304             nan     0.1000    0.0831
##      5        1.2769             nan     0.1000    0.0739
##      6        1.2297             nan     0.1000    0.0799
##      7        1.1807             nan     0.1000    0.0696
##      8        1.1388             nan     0.1000    0.0537
##      9        1.1048             nan     0.1000    0.0509
##     10        1.0724             nan     0.1000    0.0523
##     20        0.8462             nan     0.1000    0.0240
##     40        0.6072             nan     0.1000    0.0140
##     60        0.4806             nan     0.1000    0.0083
##     80        0.3881             nan     0.1000    0.0052
##    100        0.3208             nan     0.1000    0.0027
##    120        0.2702             nan     0.1000    0.0027
##    140        0.2269             nan     0.1000    0.0021
##    150        0.2108             nan     0.1000    0.0034
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2411
##      2        1.4575             nan     0.1000    0.1665
##      3        1.3514             nan     0.1000    0.1265
##      4        1.2698             nan     0.1000    0.1164
##      5        1.1980             nan     0.1000    0.0940
##      6        1.1393             nan     0.1000    0.0697
##      7        1.0935             nan     0.1000    0.0736
##      8        1.0467             nan     0.1000    0.0707
##      9        1.0032             nan     0.1000    0.0565
##     10        0.9691             nan     0.1000    0.0493
##     20        0.6958             nan     0.1000    0.0257
##     40        0.4538             nan     0.1000    0.0132
##     60        0.3274             nan     0.1000    0.0053
##     80        0.2482             nan     0.1000    0.0041
##    100        0.1935             nan     0.1000    0.0029
##    120        0.1541             nan     0.1000    0.0025
##    140        0.1241             nan     0.1000    0.0016
##    150        0.1128             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1247
##      2        1.5242             nan     0.1000    0.0895
##      3        1.4653             nan     0.1000    0.0678
##      4        1.4212             nan     0.1000    0.0529
##      5        1.3862             nan     0.1000    0.0453
##      6        1.3570             nan     0.1000    0.0446
##      7        1.3284             nan     0.1000    0.0411
##      8        1.3026             nan     0.1000    0.0364
##      9        1.2795             nan     0.1000    0.0355
##     10        1.2546             nan     0.1000    0.0318
##     20        1.0937             nan     0.1000    0.0198
##     40        0.9108             nan     0.1000    0.0101
##     60        0.7995             nan     0.1000    0.0056
##     80        0.7182             nan     0.1000    0.0039
##    100        0.6537             nan     0.1000    0.0045
##    120        0.6023             nan     0.1000    0.0043
##    140        0.5562             nan     0.1000    0.0033
##    150        0.5342             nan     0.1000    0.0031
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1913
##      2        1.4856             nan     0.1000    0.1310
##      3        1.4013             nan     0.1000    0.1055
##      4        1.3336             nan     0.1000    0.0911
##      5        1.2762             nan     0.1000    0.0764
##      6        1.2275             nan     0.1000    0.0706
##      7        1.1834             nan     0.1000    0.0568
##      8        1.1468             nan     0.1000    0.0538
##      9        1.1124             nan     0.1000    0.0466
##     10        1.0833             nan     0.1000    0.0491
##     20        0.8492             nan     0.1000    0.0254
##     40        0.6327             nan     0.1000    0.0106
##     60        0.4961             nan     0.1000    0.0084
##     80        0.3966             nan     0.1000    0.0038
##    100        0.3286             nan     0.1000    0.0045
##    120        0.2725             nan     0.1000    0.0030
##    140        0.2311             nan     0.1000    0.0028
##    150        0.2146             nan     0.1000    0.0026
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2350
##      2        1.4575             nan     0.1000    0.1635
##      3        1.3553             nan     0.1000    0.1335
##      4        1.2705             nan     0.1000    0.0951
##      5        1.2092             nan     0.1000    0.1037
##      6        1.1453             nan     0.1000    0.0793
##      7        1.0968             nan     0.1000    0.0809
##      8        1.0463             nan     0.1000    0.0687
##      9        1.0029             nan     0.1000    0.0589
##     10        0.9656             nan     0.1000    0.0566
##     20        0.6943             nan     0.1000    0.0273
##     40        0.4534             nan     0.1000    0.0148
##     60        0.3263             nan     0.1000    0.0112
##     80        0.2487             nan     0.1000    0.0069
##    100        0.1914             nan     0.1000    0.0024
##    120        0.1494             nan     0.1000    0.0021
##    140        0.1209             nan     0.1000    0.0010
##    150        0.1090             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2385
##      2        1.4584             nan     0.1000    0.1606
##      3        1.3573             nan     0.1000    0.1251
##      4        1.2801             nan     0.1000    0.1185
##      5        1.2055             nan     0.1000    0.0933
##      6        1.1468             nan     0.1000    0.0725
##      7        1.1011             nan     0.1000    0.0861
##      8        1.0483             nan     0.1000    0.0685
##      9        1.0066             nan     0.1000    0.0647
##     10        0.9668             nan     0.1000    0.0565
##     20        0.7027             nan     0.1000    0.0278
##     40        0.4539             nan     0.1000    0.0115
##     60        0.3269             nan     0.1000    0.0066
##     80        0.2508             nan     0.1000    0.0035
##    100        0.1954             nan     0.1000    0.0032
##    120        0.1569             nan     0.1000    0.0048
##    140        0.1282             nan     0.1000    0.0021
##    150        0.1141             nan     0.1000    0.0016
```

```r
pred4 <- predict(modFit4,trainDF)
confusionMatrix(pred4,trainDF$classe)  ## 99.38% 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5577   20    0    1    0
##          B    2 3756   17    2    8
##          C    0   21 3398   23    5
##          D    1    0    6 3189   21
##          E    0    0    1    1 3573
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9934          
##                  95% CI : (0.9922, 0.9945)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9917          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9995   0.9892   0.9930   0.9916   0.9906
## Specificity            0.9985   0.9982   0.9970   0.9983   0.9999
## Pos Pred Value         0.9962   0.9923   0.9858   0.9913   0.9994
## Neg Pred Value         0.9998   0.9974   0.9985   0.9984   0.9979
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1914   0.1732   0.1625   0.1821
## Detection Prevalence   0.2853   0.1929   0.1757   0.1639   0.1822
## Balanced Accuracy      0.9990   0.9937   0.9950   0.9949   0.9952
```

```r
predFinal <- predict(modFit4,testDF)
predFinal
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Appendixes
  
- System specifications : 3.1Ghz Intel Core i7, 16 GB 1867 MHz DDR3, Mac OS Sierra
- Session Info :

```
## R version 3.3.2 (2016-10-31)
## Platform: x86_64-apple-darwin13.4.0 (64-bit)
## Running under: macOS Sierra 10.12.6
## 
## locale:
## [1] en_us.UTF-8/en_us.UTF-8/en_us.UTF-8/C/en_us.UTF-8/en_us.UTF-8
## 
## attached base packages:
## [1] parallel  splines   stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
## [1] plyr_1.8.4      gbm_2.1.3       survival_2.41-3 MASS_7.3-47    
## [5] rpart_4.1-11    caret_6.0-76    lattice_0.20-35 ggplot2_2.2.1  
## [9] GGally_1.3.2   
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.10       compiler_3.3.2     RColorBrewer_1.1-2
##  [4] nloptr_1.0.4       class_7.3-14       iterators_1.0.8   
##  [7] tools_3.3.2        digest_0.6.12      lme4_1.1-13       
## [10] evaluate_0.10      tibble_1.3.0       gtable_0.2.0      
## [13] nlme_3.1-131       mgcv_1.8-17        Matrix_1.2-8      
## [16] foreach_1.4.3      DBI_0.6-1          yaml_2.1.14       
## [19] SparseM_1.77       e1071_1.6-8        dplyr_0.5.0       
## [22] stringr_1.2.0      knitr_1.17         MatrixModels_0.4-1
## [25] stats4_3.3.2       rprojroot_1.2      grid_3.3.2        
## [28] nnet_7.3-12        reshape_0.8.6      R6_2.2.0          
## [31] rmarkdown_1.6      minqa_1.2.4        reshape2_1.4.2    
## [34] car_2.1-5          magrittr_1.5       backports_1.0.5   
## [37] scales_0.4.1       codetools_0.2-15   ModelMetrics_1.1.0
## [40] htmltools_0.3.5    assertthat_0.1     pbkrtest_0.4-7    
## [43] colorspace_1.3-2   quantreg_5.33      stringi_1.1.5     
## [46] lazyeval_0.2.0     munsell_0.4.3
```
