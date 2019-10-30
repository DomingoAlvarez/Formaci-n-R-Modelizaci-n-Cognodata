#---------------------------------#
############ Packages #############
#---------------------------------#
list.of.packages <- c('ROCR','neuralnet','rpart','rattle','rpart.plot','RColorBrewer','randomForest', 'gbm','xgboost','Hmisc','corrplot','RGtk2')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

#---------------------------------#
########## Data Loading ###########
#---------------------------------#
data<-read.csv(file='C:/Users/alvarezp/Downloads/20180508 R - Modelación/Rimac_fuga_pm_train.txt'
               , sep = "\t", dec = ".", header=TRUE)

# Clean NA
data[is.na(data)]<-0

#-------------------------------------#
########## Data Preparation ###########
#-------------------------------------#
#Si queremos cambiar el nombre de alguna variable
colnames(data)[which(names(data) == "TARGET")] <- "Target"

#summary of the data to check quality
summary(data)


#split randomly the data frame 
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]

#Data normalization
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
train_ <- scaled[index,]
test_ <- scaled[-index,]

#-----------------------------------------------------------------#
########## Analisis de correlación (si fuera necesario) ###########
#-----------------------------------------------------------------#
library(Hmisc)
cor <- rcorr(as.matrix(data), type="pearson")

#correlacion
matrix_cor <- data.frame(cor$r)

#p-valor de la correlación (por encima de un umbral no es significativo)
matrix_sig <- data.frame(cor$P)

#pintar correlacion
corrplot::corrplot(cor$r, method='circle', tl.cex = 0.7 #matriz de correlacion, método de ploteo y tamaño de fuente
                   ,type = 'lower' #nos quedamos con la diagonal inferior
                   ,p.mat = cor$P, sig.level = 0.01 #tachamos con una cruz los niveles de significancia > 0.01
                   ,addCoef.col = "black",number.cex=0.5# añadimos valor del coeficiente y el tamaño
                   ,tl.col = "darkblue", tl.srt = 45#color y orientación
                   ) 

target_vs_rest <- data.frame(t(matrix_cor[1,2:ncol(matrix_cor)]))
# Tomamos el valor absoluto para seleccionar las variables más correladas
target_vs_rest[2] <- abs(target_vs_rest$Target)
target_vs_rest[1] <- rownames(target_vs_rest)
colnames(target_vs_rest) <- c('variable', 'corr_target')
rownames(target_vs_rest) <- NULL
target_vs_rest <- target_vs_rest[order(target_vs_rest$corr_target, decreasing = TRUE),]

#-----------------------------------------------------#
########## Logictic Regression as Reference ###########
#-----------------------------------------------------#

#train
glm <- glm(factor(Target) ~ ., data=train, family=binomial(link="logit") )

#results
summary(glm)

#evaluate
expl_glm<- predict(glm,test, type = "response") # prob de ser 1

#ROC curve
library(ROCR)
rocrpred_glm <- ROCR::prediction(expl_glm, test$Target) 
plot(performance(rocrpred_glm, "tpr", "fpr"), colorize = TRUE,print.cutoffs.at=seq(0,1,0.1),text.adj=c(-0.2,1.7)) 
abline(0, 1, lty = 2)
#area under the curve
performance(rocrpred_glm,"auc")@y.values

#write.csv(data.frame(expl_glm, test$Target), file='roc.txt')


#------------------------------------------------#
########## Decission Tree as Reference ###########
#------------------------------------------------#

#Train
library(rpart)
model_tree <- rpart(Target ~ .,
                    method="class", 
                    data=train,
                    control=rpart.control(minsplit=100, cp=0.001, maxdepth = 5)  )

#tree view
summary(model_tree)

plot(model_tree, uniform=TRUE, main="Target")
text(model_tree, use.n=TRUE, all=TRUE, cex=0.5)


#enhanced tree view
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(model_tree,  cex=0.6, type=2)

#evaluate
expl_tree <- predict(model_tree, test, type = "prob")

#ROC Curve
rocrpred_tree <- ROCR::prediction(expl_tree[ ,2], test$Target) 
plot(performance(rocrpred_tree, "tpr", "fpr"), colorize = TRUE,print.cutoffs.at=seq(0,1,0.1),text.adj=c(-0.2,1.7)) 
abline(0, 1, lty = 2)
#area under the curve
performance(rocrpred_tree,"auc")@y.values





#------------------------------------------------#
############ Neural Network hidden=0 #############
#------------------------------------------------#
library(neuralnet)

#esta libreria obliga a poner los nombres de las variables 1 a 1. Lo hacemos con la siguiente funcion
n <- names(train_)
f <- as.formula(paste("Target ~", paste(n[!n %in% "Target"], collapse = " + ")))

#train
model_net<- neuralnet(f,data=train_, hidden=0, threshold=0.01,stepmax=1e+5,linear.output=FALSE) 


#net view
plot(model_net,cex=0.8)

#evaluate
expl_nn <- as.data.frame(compute(model_net,test_[,!(names(test_) %in% "Target")]))

#ROC Curve
rocrpred_nn <- ROCR::prediction(expl_nn$net.result,test_$Target) 
plot(performance(rocrpred_nn, "tpr", "fpr"), colorize = TRUE,print.cutoffs.at=seq(0,1,0.1),text.adj=c(-0.2,1.7) )  
abline(0, 1, lty = 2)
#area under the curve
performance(rocrpred_nn,"auc")@y.values





#------------------------------------------------#
########### Neural Network hidden layer ##########
#------------------------------------------------#

#train
model_net_2<- neuralnet(f,
                        data=train_, hidden=c(5), threshold=0.1,stepmax=1e+6,linear.output=FALSE) 

#net view
plot(model_net_2, cex=0.8)

#evaluate
expl_nn_2 <- as.data.frame(compute(model_net_2,test_[,!(names(test_) %in% "Target")]))


#Compute MSE
MSE.nn_2 <- sum((test_$Target - expl_nn_2$net.result)^2)/nrow(test_)

#ROC Curve
rocrpred_nn_2 <- ROCR::prediction(expl_nn_2$net.result,test_$Target) 
plot(performance(rocrpred_nn_2, "tpr", "fpr"), colorize = TRUE,print.cutoffs.at=seq(0,1,0.1),text.adj=c(-0.2,1.7)) 
abline(0, 1, lty = 2)
#area under the curve
performance(rocrpred_nn_2,"auc")@y.values


#--------------------------------------------------#
#################  Random Forest  ##################
#--------------------------------------------------#
library(randomForest)

model_rf <- randomForest(as.factor(Target) ~ .,
                         data=train_,
                         importance=TRUE,
                         nodesize=25,
                         ntree=100)

#variables importantes
varImpPlot(model_rf,type=2)

#evaluate
expl_rf <- predict(model_rf, test_,"prob")

#ROC Curve
rocrpred_rf <- ROCR::prediction(expl_rf[ ,2], test_$Target) 
plot(performance(rocrpred_rf, "tpr", "fpr"), colorize = TRUE,print.cutoffs.at=seq(0,1,0.1),text.adj=c(-0.2,1.7)) 
abline(0, 1, lty = 2)
#area under the curve
performance(rocrpred_rf,"auc")@y.values


#--------------------------------------------------#
################  Gradient Booting #################
#--------------------------------------------------#
library(gbm)

model_gbm <- gbm(Target ~ .,
                 data=train_,
                 dist="bernoulli", 
                 n.tree = 2000,
                 shrinkage = 0.01, #es como el learning rate
                 train.fraction = 0.5 #fraccion que se usará para hacer el train, el resto es para ajuste
                 )

# gbm.perf returns estimated best number of trees.
gbm.perf(model_gbm)
summary(model_gbm)

expl_gbm <- predict(model_gbm, test_,type="response")
rocrpred_gbm <- ROCR::prediction(expl_gbm,  test_$Target) 
plot(performance(rocrpred_gbm, "tpr", "fpr"), colorize = TRUE,print.cutoffs.at=seq(0,1,0.1),text.adj=c(-0.2,1.7)) 
abline(0, 1, lty = 2)
#area under the curve
performance(rocrpred_gbm,"auc")@y.values


#---------------------------------#
##########   XGboost   ############
#---------------------------------#
library(xgboost)

#en esta función todas las entradas deben ser matrices
model_xgboost <- xgboost::xgboost(data = data.matrix(train_[ , -which(names(train_) %in% c("Target"))]), label = data.matrix(train_$Target), 
                         max_depth = 6, #profundidad de los árboles
                         eta = 0.3, #learning_rate
                         nthread = 4, 
                         nrounds = 20, #iteraciones
                         objective = "binary:logistic")

expl_xgboost <- predict(model_xgboost, data.matrix(test_[ , -which(names(train_) %in% c("Target"))]))

#curva ROC
rocrpred_xgboost <- ROCR::prediction(expl_xgboost,test_$Target) 
plot(performance(rocrpred_xgboost, "tpr", "fpr"), colorize = TRUE,print.cutoffs.at=seq(0,1,0.1),text.adj=c(-0.2,1.7)) 
abline(0, 1, lty = 2)
#area bajo la curva
performance(rocrpred_xgboost,"auc")


#--------------------------------------------#
################ Comparison ##################
#--------------------------------------------#
plot(performance(rocrpred_tree, "tpr", "fpr"), col=2, main="ROC Curve: Comparison between Algorithms")
plot(performance(rocrpred_glm, "tpr", "fpr"), col=3, add=TRUE)
plot(performance(rocrpred_nn, "tpr", "fpr"), col=4, add=TRUE)
plot(performance(rocrpred_nn_2, "tpr", "fpr"), col=5, add=TRUE)
plot(performance(rocrpred_rf, "tpr", "fpr"), col=6, add=TRUE)
plot(performance(rocrpred_gbm, "tpr", "fpr"), col=7, add=TRUE)
plot(performance(rocrpred_xgboost, "tpr", "fpr"), col=8, add=TRUE)
abline(0, 1, lty = 2)
legend(0.6, 0.45, c('Tree','GLM', "NN hidden=0", "NN hidden=5", "RandomForest", "Gradient Boosting", "XGboost"), 2:8, cex=0.8)

#Area under the curve
auc_tree<-performance(rocrpred_tree,"auc")
auc_glm<-performance(rocrpred_glm,"auc")
auc_nn<-performance(rocrpred_nn,"auc")
auc_nn2<-performance(rocrpred_nn_2,"auc")
auc_rf<-performance(rocrpred_rf,"auc")
auc_gbm<-performance(rocrpred_gbm,"auc")
auc_xgboost<-performance(rocrpred_xgboost,"auc")

results<-as.data.frame(cbind(auc_tree=as.numeric(auc_tree@y.values),
                             auc_glm=as.numeric(auc_glm@y.values),
                             auc_nn=as.numeric(auc_nn@y.values),
                             auc_nn2=as.numeric(auc_nn2@y.values),
                             auc_rf=as.numeric(auc_rf@y.values),
                             auc_gbm=as.numeric(auc_gbm@y.values),
                             auc_xgboost=as.numeric(auc_xgboost@y.values)
                             ))


#Compute MSE
MSE_tree <- sum((test_$Target - expl_tree[,2])^2)/nrow(test_) 
MSE_glm <- sum((test_$Target - expl_glm)^2)/nrow(test_) 
MSE_nn <- sum((test_$Target - expl_nn$net.result)^2)/nrow(test_) 
MSE_nn2 <- sum((test_$Target - expl_nn_2$net.result)^2)/nrow(test_) 
MSE_rf <- sum((test_$Target - expl_rf[,2])^2)/nrow(test_) 
MSE_gbm <- sum((test_$Target - expl_gbm)^2)/nrow(test_)
MSE_xgboost <- sum((test_$Target - expl_xgboost)^2)/nrow(test_)

mse <-as.data.frame(cbind(tree=MSE_tree,
                          glm=MSE_glm,
                          nn=MSE_nn,
                          nn2=MSE_nn2,
                          rf=MSE_rf,
                          gbm=MSE_gbm,
                          xgboost=MSE_xgboost))


# Probabilidades del test
Probabilidades <- as.data.frame(cbind(glm=expl_glm,
                                      tree=expl_tree[ ,2],
                                      net=expl_nn$net.result,
                                      net2=expl_nn_2$net.result,
                                      rf=expl_rf[ ,2],
                                      gbm=expl_gbm,
                                      xgboost=expl_xgboost,
                                      Target=test$Target)
                                )

#ordenamos según probabilidad del RF
Probabilidades <- Probabilidades[order(Probabilidades$rf,decreasing = TRUE),]                                 
write.table(Probabilidades, "c:/mydata_rimac.txt", sep="\t",row.names = F)




