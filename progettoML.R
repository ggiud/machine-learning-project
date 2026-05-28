saturn = readRDS("Saturn_dataset.RDS")
# plot(saturn)
# divisione chiara delle classi in 2D per le prime due dimesioni:
plot(saturn$feature.1, saturn$feature.2, col=(saturn$class+2))
library(EMCluster)
library(caret)
library(tidyverse)
library(randomForest)


### K-MEANS---------------------------------------------------------------------
km.res = kmeans(saturn[,-9], 2, iter.max=100)
plot(saturn$feature.1, saturn$feature.2, col=km.res$cluster) # NO
table(km.res$cluster, saturn$class)

# k-means con riscalaggio classico:
km_std.res = kmeans(scale(saturn[,-9]), 2, iter.max=100)
plot(saturn$feature.1, saturn$feature.2, col=km_std.res$cluster) # NO
table(km_std.res$cluster, saturn$class)

# k-means con riscalaggio mediante criterio min-max:
saturn_mms = apply(saturn, 2, function(x) (x-min(x))/(max(x)-min(x)))
km_std.res = kmeans(saturn_mms[,-9], 2, iter.max=100)
plot(saturn$feature.1, saturn$feature.2, col=km_std.res$cluster) # NO
(mat1 = table(km_std.res$cluster, saturn$class))
(acc1 = sum(diag(mat1))/nrow(saturn))


### HIERARCHICAL CLUSTERING-----------------------------------------------------
D = dist(saturn_mms[,-9])
h.res = hclust(D, method="complete")
cluster = cutree(h.res, k=2)
plot(saturn$feature.1, saturn$feature.2, col=cluster)
(mat2 = table(cluster, saturn$class))
(acc2 = sum(diag(mat2))/nrow(saturn))


### EM Clustering---------------------------------------------------------------
k = 2
embj = init.EM(saturn[,-9], nclass=k)
em.res = emcluster(saturn[,-9], embj, assign.class=T)
plot(saturn$feature.1, saturn$feature.2, col=em.res$class)
(mat3 = table(em.res$class, saturn$class))
(acc3 = sum(diag(mat3))/nrow(saturn)) # PERFETTO

# train/test:
set.seed(1)
idx = sample(1:nrow(saturn), 0.7*nrow(saturn))
train = saturn[idx, -9]
test = saturn[-idx, -9]

embj = init.EM(train, nclass=2)
em.res = emcluster(train, embj, assign.class=TRUE)

# Predizioni sul test:
pred = assign.class(test, em.res)
table(pred$class, saturn[-idx, "class"])



### KNN-------------------------------------------------------------------------
knn_k = 10:40
set.seed(123)
# lascio un 10% di test-set per la valutazione finale:
ixs = createDataPartition(saturn$class, times=1, p=0.9)
dataset = saturn[ixs$Resample1,]
testset = saturn[-ixs$Resample1,]

featRanges = apply(dataset[, -ncol(dataset)], 2, range)
for(j in 1:ncol(featRanges)) { # applico standardizzazione al dataset e testset con criterio max-min (dataset)
  dataset[, j] = (dataset[, j]-featRanges[1, j])/diff(featRanges[, j])
  testset[, j] = (testset[, j]-featRanges[1, j])/diff(featRanges[, j])
}

## hold-out:
# divido il dataset rimanente in training e validation:
trnIxs = createDataPartition(dataset$class, times=1, p=0.8)

# ciclo per calcolare err emp e gen per tutti i k:
empiricalErr = generalizationErr = predErr = numeric(length(knn_k))
for(i in 1:length(knn_k)) {
  preds = knn3Train(train=dataset[trnIxs$Resample1, -ncol(dataset)], # sul training
                    test=dataset[trnIxs$Resample1, -ncol(dataset)],
                    cl=dataset$class[trnIxs$Resample1], k=knn_k[i], prob=F, use.all=T)
  empiricalErr[i] = mean(dataset$class[trnIxs$Resample1] != preds)
  preds = knn3Train(train=dataset[trnIxs$Resample1, -ncol(dataset)], # sul validation
                    test=dataset[-trnIxs$Resample1, -ncol(dataset)],
                    cl=dataset$class[trnIxs$Resample1], k=knn_k[i], prob=F, use.all=T)
  generalizationErr[i] = mean(dataset$class[-trnIxs$Resample1] != preds)
}

# grafico del confronto degli err emp e gen per tutti i k:
plot(knn_k, empiricalErr, type="o")
lines(knn_k, generalizationErr, type="o", col="red")

# k=25:
empiricalErr[21]
generalizationErr[21]
cat("> Empirical error on dataset:",empiricalErr[21],
    "(Acc=",round(100*mean(1-empiricalErr[21]), 2),"%)\n")
cat("> Generalization error on dataset:",generalizationErr[21],
    "(Acc=",round(100*mean(1-generalizationErr[21]), 2),"%)\n")

# scelto k=25, calcolo previsioni ed errore di previsione:
preds = knn3Train(train=dataset[, -ncol(dataset)], # applico KNN sul dataset
                  test=testset[, -ncol(dataset)], # e predico sul test set
                  cl=dataset$class, k=25, prob=F, use.all=T)
predErr = mean(testset$class != preds)

cat("> Prediction error on test set:",predErr,
    "(Acc=",round(100*mean(1-predErr), 2),"%)\n")


## decido di optare per un metodo diverso per la scelta del valore di knn_k, ovvero mediante k-fold CV:
knn_k  = 10:40

set.seed(123)
ixs = createFolds(dataset$class, k = 10, list = T)

trnErrMeans = valErrMeans = numeric(length(knn_k))

for(i in 1:length(knn_k)) {
  trnErrs = valErrs = numeric()
  for(k in 1:length(ixs)) {
    
    valFold = dataset[ixs[[k]], ]
    trnFold = dataset[-ixs[[k]], ]
    
    preds = knn3Train(train = trnFold[, -ncol(trnFold)],
                       test = trnFold[, -ncol(trnFold)],
                       cl = trnFold$class, k = knn_k[i], prob = F, use.all = T)
    trnErrs[k] = mean(trnFold$class != preds) 
    
    preds = knn3Train(train = trnFold[, -ncol(trnFold)],
                       test = valFold[, -ncol(valFold)],
                       cl = trnFold$class, k = knn_k[i], prob = F, use.all = T)
    valErrs[k] = mean(valFold$class != preds) 
  }
  
  trnErrMeans[i] = mean(trnErrs)
  valErrMeans[i] = mean(valErrs)
  
}

# grafico ggplot
df <- data.frame(
  k = knn_k,
  Empirical = trnErrMeans,
  Generalization = valErrMeans
) %>%
  pivot_longer(cols = c(Empirical, Generalization),
               names_to = "ErrorType",
               values_to = "Error")

ggplot(df, aes(x = k, y = Error, color = ErrorType)) +
  geom_line(size = 1) +       
  geom_segment(x = 20, xend = 20, y = 0.035, yend = 0.085,
               color = "black", linetype = "dashed", size = 0.8) +
  geom_point(size = 3, shape = 19) +  
  scale_color_manual(values = c("red", "green")) +
  geom_hline(yintercept = 0.1, linetype = "dashed", size = 1, col = 'grey80') +
  labs(x = "k", y = "Error", color = "Error Type", 
       title = "Empirical vs Generalization Error") +
  ylim(c(0.03, 0.16)) +
  theme_minimal()+
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    legend.position = c(0.2, 0.8), 
    legend.text = element_text(size = 13),      
    legend.title = element_text(size = 14),
    legend.background = element_rect(fill = "white"))


plot(knn_k, trnErrMeans, type = "o", lwd = 2, pch = 19, col = "red", 
     ylim = c(0.03, 0.17))
lines(knn_k, valErrMeans, type = "o", pch = 19, lwd = 2,
      xlab = "k", ylab = "Error", col = "green")
legend("topleft", legend = c("Empirical Error", "Generalization Error"),
       col = c("red", "green"), lty = 1, lwd = 2, pch = 19)
abline(h = 0.1, lwd = 2, lty = 2)

plot(10:40, valErrMeans - trnErrMeans, col = 'blue', lwd = 2, type = 'o')
abline(v = 25, col = 'red')

# quindi procedo con le stime finali...
preds.fin = knn3Train(train = dataset[, -ncol(dataset)], 
                       test = testset[, -ncol(dataset)], cl = dataset$class,
                       k = 20, prob = F, use.all = T)
(predsErr = mean(as.factor(preds.fin) != testset$class))

### SVM-------------------------------------------------------------------------
library(e1071)
mdl = svm(x=saturn[,1:(ncol(saturn)-1)], y=saturn[,ncol(saturn)],
          type="C-classification", kernel="radial", scale=T)
preds = predict(mdl, saturn[,1:(ncol(saturn)-1)])
(confM = table(preds, saturn[,ncol(saturn)]))
plot(saturn$feature.1, saturn$feature.2, col=preds)
(acc4 = sum(diag(confM))/nrow(saturn))
# determino il valore degli iperparametri mediante BO:
library(mlr)
library(mlrMBO)
set.seed(349)
par.set = makeParamSet(makeNumericParam("cost", lower=-2, upper=4, trafo=function(x) 10^x),
                       makeNumericParam("gamma", lower=-4, upper=1, trafo=function(x) 10^x))
ctrl = makeMBOControl()
ctrl = setMBOControlTermination(ctrl, iters=15)
tune.ctrl = makeTuneControlMBO(mbo.control=ctrl)
saturn$class = as.factor(saturn$class)
task = makeClassifTask(data=saturn, target="class")
cv3 = makeResampleDesc("CV", iters=5)
run = tuneParams(makeLearner("classif.svm"), task, cv3, measures=acc, par.set=par.set,
                 control=tune.ctrl)
ys = getOptPathY(run$opt.path)
plot(ys, type="o", pch=19, lwd=3, col="grey")
lines(cummax(ys), type="s", lwd=3, col="green")
run$x # parametri ottimali da usare per il modello
# quindi...
mdl.opt = svm(x=saturn[,1:(ncol(saturn)-1)], y=saturn[,ncol(saturn)],
              type="C-classification", kernel="radial", scale=T,
              cost=run$x$cost, gamma=run$x$gamma)
preds = predict(mdl.opt, saturn[,1:(ncol(saturn)-1)])
confM = table(preds, saturn[, ncol(saturn)]); print(confM)
plot(saturn$feature.1, saturn$feature.2, col=preds)
cat("Empirical error:",round(100*(1-(sum(diag(confM))/sum(confM))), 2),"%\n")
cat("number of svm:",mdl.opt$nSV,"(",round(100*mdl.opt$tot.nSV/nrow(saturn), 2),"%)\n")
(acc4 = sum(diag(confM))/nrow(saturn))



C_values = c(20, 21, 23.93538, 25) # weight dealing the tradeoff between the margin and the classif-error
g_values = c(0.01, 0.02190278, 0.03)
folds = createFolds(y=saturn[, ncol(saturn)], k=10, returnTrain=T, list=T)
empErr = genErr = percSVs = numeric()

for (c in C_values){
  for (g in g_values){
    for (i in 1:length(folds)){
      trn_x = saturn[folds[[i]], -ncol(saturn)] # data
      trn_y = saturn[folds[[i]], ncol(saturn)] # labels
      val_x = saturn[-folds[[i]], -ncol(saturn)]
      val_y = saturn[-folds[[i]], ncol(saturn)]
      mdl = svm(x=trn_x, y=trn_y, type="C-classification",
                kernel="radial", scale=T, cost=c, gamma=g)
      preds = predict(mdl, trn_x)
      empErr[i] = mean(preds != trn_y)
      percSVs[i] = mdl$tot.nSV/length(folds[[i]])
      preds = predict(mdl, val_x)
      genErr[i] = mean(preds != val_y)
    }
    cat("C =", c, "gamma=", g, "; avgEmpErr:", round(mean(empErr), 4),
        "; avgGenErr:", round(mean(genErr), 4), "; %SVs:", round(mean(percSVs), 4), "\n")
  }
}







# verifica dei risultati mediante training e test (validation) set con BO:
set.seed(567)
n = nrow(saturn)
train.idx = sample(1:n, 0.9*n)
train = saturn[train.idx, ]
test  = saturn[-train.idx, ]
saturn$class = as.factor(saturn$class)
task.train = makeClassifTask(data=train, target=names(saturn)[ncol(saturn)])
cv3 = makeResampleDesc("CV", iters=5)
par.set = makeParamSet(
  makeNumericParam("cost",  lower=-2, upper=4, trafo=function(x) 10^x),
  makeNumericParam("gamma", lower=-4, upper=1, trafo=function(x) 10^x))
ctrl = makeMBOControl()
ctrl = setMBOControlTermination(ctrl, iters=15)
tune.ctrl = makeTuneControlMBO(mbo.control=ctrl)
run = tuneParams(makeLearner("classif.svm"), task=task.train, resampling=cv3,
                 measures=acc, par.set=par.set, control=tune.ctrl)
run$x
mdl = svm(x=train[, -ncol(train)], y=train[, ncol(train)], type="C-classification",
          kernel="radial", scale=T, cost=run$x$cost, gamma=run$x$gamma)
preds.test = predict(mdl, test[, -ncol(test)])
confM.test = table(preds.test, test[, ncol(test)]); print(confM.test)
cat("Empirical error:",round(100*(1-(sum(diag(confM.test))/sum(confM.test))), 2),"%\n")
cat("number of svm:",mdl$nSV,"(",round(100*mdl$tot.nSV/nrow(train), 2),"%)\n")
(acc.test = sum(diag(confM.test))/sum(confM.test))

### random forest---------------------------------------------------------------
set.seed(12345)
saturn$class = as.factor(saturn$class)
tuneRF(saturn[,-9], saturn$class) # per numero di variabili da usare
rf <- randomForest(class ~ ., data = saturn, ntree = 500, mtry = 2)
print(rf)
plot(rf)
varImpPlot(rf)


# PLOT RF
err <- data.frame(
  Trees = 1:nrow(rf$err.rate),
  rf$err.rate
)

err_long <- pivot_longer(err, cols = -Trees, names_to = "Type", 
                         values_to = "Error")

ggplot(err_long, aes(x = Trees, y = Error, color = Type)) +
  geom_line(size = 1) +
  labs(title = "Random Forest Errors",
       x = "Number of Trees",
       y = "Error",
       color = "Error Type") +
  scale_color_manual(values = c("OOB" = "red",
                                "X.1" = "green", 
                                "X1" = "blue"),
                     labels = c("OOB" = "OOB totale",
                                "X.1"  = "Errore classe -1",
                                "X1"   = "Errore classe 1")) +
  ylim(c(0, 0.04))+
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = c(0.8, 0.8),
        legend.background = element_rect(fill = "white"))
