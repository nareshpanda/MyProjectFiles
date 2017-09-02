## Code amended on 28th Jan, 2017
## Problem: Education Analytics: Predict which child is likely to drop-out from school
# Ref:    https://github.com/ZhouFang928/EduDP
#         https://github.com/dalpozz/unbalanced
# See file studentDropIndia_nnet_noBalance.R for unbalanced data 
#     (not properly corrected, but works)
# File folder: C:\Users\ashok\OneDrive\Documents\education_analytics\final

## Objectives
##          1. Learn to impute data
##          2. Learning to deal with unbalanced data. (SMOTE)
##          3. Learn to use nnet
##          4. Learn to tune parameters
##          5. Learn performance metrics (ROC/AUC)
##          6. Perform the experiment as here with unbalanced data also
##          7. How to specify cost matrix in C50 (see note at end of code)


########################## BEGIN #################################

## 1. Delete past objects and call libraries-------
#     Load required packages from local library into R.
rm(list=ls()) ; gc()
library(magrittr)     # Data pipelines: %>% %T>% %<>%.
library(unbalanced)   # Balancing using ubSMOTE Another library is=> DMwR
library(rpart)        # Model: decision tree.
library(rattle)       # Draw fancyRpartPlot().
library(rpart.plot)   # Draw fancyRpartPlot().
library(randomForest) # For imputing using na.roughfix()   
library(e1071)        # For parameter tuning tune.nnet()
library(nnet)         # Model: neural network.
library(caret)        # createDataPartition(), confusionMatrix, featurePlot()
library(pROC)         # Draw roc graph & calculate AUC
library(stringi)      # For %s+% (string concatenate)


### 2. Identify the dataset to load----
setwd("C:\\Users\\ashok\\OneDrive\\Documents\\education_analytics\\data")
dir()
ds<-read.csv("studentDropIndia_20161215.csv", header=TRUE)

# 3 Observe education data
head(ds)
str(ds)

# 3.1 Convert column 'internet' from logical to factor
#     Rest of the fields are properly structured
ds$internet<-as.factor(ds$internet)

# 3.2 Remove missing values
# 3.21  Get names of numeric columns
#         'numc' will contain only numeric column names 
#           (not data). Exclude cols 2 & 12 that are IDs (even though numeric)
numc<-ds[,-c(2,12)] %>%                         # All columns but 2 and 12
                    sapply( is.numeric) %>%     # Output is TRUE, FALSE...
                    which() %>%                 # Output is col index wherever input is TRUE
                    names()                     # Outputs col names for an input index
numc
# 3.3 Impute missing values. Else nnet will not work
#       na.roughhfix() fixes numerical matrix by median (if, numeric)/mode(if, categorical)
#         na.roughfix() is a function from randomForest()
ds[,numc] %<>% na.roughfix()     
ds[,-c(2,12)] %>%  is.na() %>% sum()  

# 3.4 Initialise random numbers for repeatable results.
set.seed(123)

# 4. Partition the full dataset into two. Stratified sampling. 
trainInd<-createDataPartition(ds$continue_drop, p=0.7,list=FALSE)
valid<- ds[-trainInd,]

# 5. train data: But separate predictors and target
#    1: continue_drop, 2: student_id, 12: school_id
X<-ds[trainInd, -c(1,2,12)]    
#    Get y as a factor NOT 1,2 as now but 0 and 1
y<-as.factor(as.numeric(ds[trainInd , 1]) -1) # 1st col is target

# 6. Balance train dataset now
#  ubSMOTE implements SMOTE (Synthetic Minority Oversampling Technique)
#     which oversamples the minority class
#      by generating synthetic minority examples
#       in the neighborhood of observed ones.
#     Ref: https://github.com/dalpozz/unbalanced
# perc.over: per.over/100, number of new instances generated
#            for each rare instance.
# k: Number of neighbours to consider as the pool
#    from where the new examples are generated
# perc.under: perc.under/100 number of majority class instances
#                selected randomly for each 
#                   smoted observation
# 6.1 
dim(X)
b_data <- ubSMOTE(X = X, Y = y,   # Also y be a vector not a dataframe
                 perc.over=200,   #  200/100 = 2 instances generated for every rare instance
                 perc.under=500,  #  500/100 = 5 instances selected for every smoted observation
                 k=3,
                 verbose=TRUE) 
# 6.2 ubSMOTE returns balanced data frame in b_data$X
#      and corresponding class values, as vector in b_data$Y
#       Return value 'b_data' itself is a list-structure
#     So complete and balanced train data is:
traindata <- cbind(b_data$X, continue_drop = b_data$Y)

# 6.3
dim(traindata)  # Less data

# 6.4 Check the dropping-out proportion
ds %>% count(continue_drop)  %>% mutate(n/nrow(ds))
table(traindata$continue_drop)/nrow(traindata) # Now

###### 7. Data Visualization-----------
# We will perform Visualization of balanced data

# 7.1 Obverve data
#     We will not preprocess data. Preproceesing is essential
#     For nnet()
View(traindata)

# 7.2 Plot scatterplots of features. Is there a pattern? 
#     See 'lattice' help to understand parameters in detail
#     featurePlot() is a wrapper for different lattice
#      plots to visualize data. It is a shortcut for graphs
#      Interpretation:
#        In the graphs, pink points show a pattern instead of 
#           being spread all around randomly
# Scatterplots in pairs
featurePlot(x = traindata[, 3:6],        # Marks & scienceteacher
            y = traindata$continue_drop, # Develop relationship with y
            plot = "pairs",              # Plot in pairs
            auto.key = TRUE)

#     (auto.key is typically used to automatically produce suitable
#     legend in conjunction  with  grouping-variable.)

# 7.3 Scatterplot Matrix with Ellipses. Ellipses encircle pink area
featurePlot(x = traindata[, 3:6],    # featurePlot() is a caret function
            y = traindata$continue_drop, 
            plot = "ellipse",
            ## Add a key (legend) at the top
            auto.key = list(columns = 3))

# 7.4 Overlayed Density Plots
#     Interpretation: 
#      Density plots for cases 1 and 0 almost overlap for englishmarks
#       But not for others

featurePlot(x = traindata[, 3:6], 
            y = traindata$continue_drop,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier and broader
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, # Adjusts curve smoothness
            pch = "|",    # Point character at the bottom of graph to show pt density
            layout = c(4, 1),   # Four columns
            auto.key = list(columns = 3))

# ('relation' determines how axis limits are calculated for each panel.
#   Possible values are "same" (default), "free" and "sliced".
#   For relation="same", same limits, usually large enough to encompass
#   all the data, are used for all the panels.  For relation="free",
#   limits for each  panel  is  determined  by  just  the  points  in
#   that  panel.)

# 7.5 Box plot of the features
#     Interpretation:
#        Median difference for cases of 1 and 0 for englishmarks are not as
#        distict as other marks
featurePlot(x = traindata[, 3:6], 
            y = traindata$continue_drop, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(4,1 ), 
            auto.key = list(columns = 2))


#################### Predictive Modeling #####################

### 8. Decision tree model with rpart-------
# 8.1 We use rpart and not C50
ctrl <- rpart.control(maxdepth=3)  # Max depth of tree
model.rp <- rpart(
                      continue_drop ~ .,
                      traindata,
                      control=ctrl
                      )


# 8.2 Model and its plot
model.rp
fancyRpartPlot(model.rp)  # A rattle function

## 9. Make predictions for validation dataset
# 9.1  First predict class values
#      1: continue_drop, 2: student_id, 12: school_id
class_predictions <- predict(model.rp, valid[, -c(1,2,12) ], type="class")
# What were actual values. Make actual of form 1,1,0,1 and as factor
actual<-as.factor(as.numeric(valid[,1])-1)  

## 9.2. Evaluate results. Print accuracy, kappa, recall, precision and F value
confusionMatrix(class_predictions, actual,
                positive = "1",                 # Positive class is '1'
                dnn=c("predictions","actual"),  # Dimenson headings
                mode="prec_recall")    # Print precision and recall as well

# 9.3 Predict probabilities
prob_predictions <- predict(model.rp, valid[, -c(1,2,12) ], type="prob")
head(prob_predictions)

# 10. Draw ROC graph using predicted probabilities
df_roc<-roc(actual,prob_predictions[,2])
plot(df_roc,main="AUC" %s+% df_roc$auc)


###### 11. Neural network model---------
# We use nnet package
#  But without preProcessing data
Sys.time()
# 11.1 List sampling/training parameters in a control list 
tc <- tune.control(sampling = c("cross"),   # Sampling scheme; cross against bootstrap & fixed
                   cross = 5,               # Number of partitions for cross-validation
                   nrepeat=1,               # How often training will be repeated
                   best.model = TRUE,       # Return the best model
                   performances = TRUE,     # Return the performance results of all tries
                   error.fun = NULL         # NULL => Minimize missclassification error, if target is discrete
                  )
                               
# 11.2 Modeling and tuning. Takes about 500 seconds
#      Model capable of tuning only two parameters:
#                         a. size: No of neurons in hidden layer
#                   and,  b. decay: Learning rate 
system.time(
modelTune <- tune.nnet(
                        continue_drop ~ .,         
                        data=traindata, 
                        size=c(3, 6, 7, 8,  10),      # Test size: 3, 6, 7, 8, 10
                        decay=c(5*10^(-5:-1), 0.25 ), # = c(0.00005 0.00050 0.00500 0.05000 0.50000 0.25000)
                        rang=0.6 ,   # Can't be tuned. Default=0.5. Start wts set between (-rang, +rang)
                        maxit=200,   # can't be tuned
                        tunecontrol = tc, # Type of sampling, CV, bootstrap, fixed etc
                        trace = TRUE  # Be verbose
                     )
)
Sys.time()
# 11.3 Tuning parameter-wise performanc results. 
#      A data frame of performance results
modelTune$performances

# 11.4 What were the best parameter values?
s<-modelTune$best.parameters[1]  # Size
s
d<-modelTune$best.parameters[2]  # Decay
d

# 12. Substitute these values in final model
model.net <- nnet(
                  continue_drop ~ .,
                  data=traindata,
                  size= s$size, 
                  decay= d$decay, 
                  rang=0.6,
                  trace=TRUE,
                  maxit=200
                  )

# 12.1 Check the model information
model.net

# 13. Make predictions for validation dataset
#      type="class" gives class_values (1 or 0 NOT continue/drop)
#        even though valid[,1] is continue/drop AND not 0/1
class_pred <- predict(model.net, valid[,-1], type="class")

# 13.1 type="raw" gives values at the output. This can be interpreted
#           as probabilities
prob_pred <- predict(model.net, valid[,-1], type="raw")
head(prob_pred)  # Probabilities of being 1. Only one column

# 14. Evaluate nnet model
# 14.1  First get actual values from validation dataset
(actual<-as.numeric(valid[,1]))    # Values are 1,2,1,1...
(actual<- as.factor(actual -1))    # Values are 0,1,0,0...
## OR
# 14.2 Map continue/drop to 0/1
actual<-ifelse(valid$continue_drop == "continue", 0, 1)
table(actual)  # Check 

## 14.2 Evaluate results. Print accuracy, kappa, recall, precision and F value
confusionMatrix(class_pred, actual,
                positive = "1",           # More interest in its accuracy. Change it to 0!!
                dnn=c("predictions","actual"),
                mode="prec_recall")

# 15. Draw ROC graph of TPR vs FPR
# See the explanation of ROC graphs below
# 15.1
df_roc<-roc(as.factor(valid$continue_drop) , prob_pred[,1],   direction =  "auto")
plot(df_roc,main="AUC" %s+% df_roc$auc)

# 15.2
df_roc<-roc(as.factor(valid$continue_drop) , prob_pred[,1],   direction =   ">")
plot(df_roc,main="AUC" %s+% df_roc$auc)

# 15.3
df_roc<-roc(as.factor(valid$continue_drop) , 1-prob_pred[,1], direction =   "<")
plot(df_roc,main="AUC" %s+% df_roc$auc)
########### FINISH #######################

#Behaviour of pROC in drawing ROC graph:
#==========================================
# Question raised in the class was why both the folllowing give same graph:
df_roc<-roc(as.factor(valid$continue_drop) , prob_pred[,1])
df_roc<-roc(as.factor(valid$continue_drop) , 1-prob_pred[,1])

# The reason is in the first case '1' is treated as positive class
#   and in the second case '0' is treated as positive class (auto).
# Detailed reasons are given below
#1.
# ROC graph is constructed for binary outputs. One of the two groups
# is treated as 'control' or healthy group and the other is treated as
# the 'cases' (or diseased group). Cases are generally labeled as '1'
#
#2.
# The simple formulas for constructing ROC graph is:
#   roc(response, predictor,  direction=c("auto", "<", ">"))
# The order or variables are: response (a factor variable) and values
# of predictors. It is expected that levels in response will be coded
# and ordered as: Controls, Cases.

# For example: 
#   > levels(as.factor(valid$continue_drop))
# [1] "0" "1"
# Then, "0" is for control group and "1" is for cases group.
#
# 3.
# The help for 'direction' states as follows (threshold > or <): 
#  direction:	
#  in which direction to make the comparison? "auto" (default): automatically
#  define in which group the median is higher and take the direction accordingly.
#  ">": if the predictor values for the control group are higher than the values
#  of the case group (controls > t >= cases).
#  "<": if the predictor values for the control group are lower or equal than
#  the values of the case group (controls < t <= cases).
#
# 4.
# The following all three draw graphs appropriately:
#   df_roc<-roc(as.factor(valid$continue_drop) , prob_pred[,1],   direction =   ">")
#   df_roc<-roc(as.factor(valid$continue_drop) , 1-prob_pred[,1], direction =   "<")
#   df_roc<-roc(as.factor(valid$continue_drop) , prob_pred[,1],   direction =  "auto")
#   plot(df_roc,main="AUC" %s+% df_roc$auc)
#
#5.
# Example (note the direction of threshold):
#        data		values
#         0		  0.1		    If threshold < 0.5, it is 0
#         0		  0.1
#         0		  0.1
#
#         0		  0.9		    If threshold > 0.5  it is 0
#         0		  0.9
#         0		  0.9

########## Cost Matrix ################

# For unbalanced data, in C50, one can specify a cost matrix.
# Cost matrix in C50
# Ref :http://www.patricklamle.com/Tutorials/Decision%20tree%20R/Decision%20trees%20in%20R%20using%20C50.html#costmatrix
#     : http://stackoverflow.com/questions/18206091/how-to-set-costs-matrix-for-c5-0-package-in-r
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2)
error_cost
# NB : Order of the cost matrix is following:
#  The cost matrix should by CxC, where C is number
#   of classes. Diagonal elements are ignored. Columns
#    should correspond to the true classes and rows are
#     predicted classes.
# In above case : the # for the columns and row are based
#  on the factors of the variable default where "no=1"
#    and "yes=2". For instance, the "4" on the table means
#     True class is "yes" (column 2) and Predicted class is
#     "no" (row 1).

