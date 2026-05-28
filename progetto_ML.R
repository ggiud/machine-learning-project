rm(list=ls()); graphics.off(); cat("\014")

# PROGETTO MACHINE LEARNING ------------------------------------------------
# ADEZIO, GUZZINATI, PERINA ------------------------------------------------

### LIBRARY ----------------------------------------------------------------
library(EMCluster)
library(caret)
library(ggplot2)
library(plotly)
library(e1071)
library(mlr)
library(mlrMBO)
library(randomForest)
library(tidyverse)
library(skimr)
library(GGally)
library(ggcorrplot)
library(gridExtra)
library(magick)
library(webshot2)
library(htmlwidgets)

### DATASET ----------------------------------------------------------------
saturn <- readRDS("Saturn_dataset.RDS")
plot(saturn)
plot(saturn$feature.1, saturn$feature.2, col = (saturn$class+3), pch = 19)
saturn$class <- as.factor(saturn$class)

# ANALISI VARIABILI --------------------------------------------------------
skim(saturn)

summary(saturn)

# GGPAIRS
class_colors <- c("-1" = "red", "1" = "goldenrod1")
class_labels <- c("-1" = "Class -1", "1" = "Class 1")

ggpairs(saturn, columns = 1:8, aes(color = class),        
        diag = list(continuous = wrap("barDiag", col = 'black', 
                                      bins = 20, alpha = 0.8, position = "stack")),
        lower = list(continuous = wrap("points", alpha = 0.5, size = 0.8)),
        upper = list(continuous = wrap("points", alpha = 0.5, size = 0.8))) +
    scale_color_manual(values = c('red', 'goldenrod1'), 
                       labels = c('Classe A', 'Classe B')) +
    scale_fill_manual(values = c('red', 'goldenrod1'),
                      labels = c('Classe A', 'Classe B')) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          legend.position = "right",
          legend.background = element_rect(fill = "white"))

(p <- ggpairs(saturn, columns = 1:8, aes(color = class),
              diag = list(continuous = wrap("barDiag", aes(fill = class), col = 'black', bins = 20, alpha = 0.8, position = "stack")),
              lower = list(continuous = wrap("points", alpha = 0.5, size = 0.8)),
              upper = list(continuous = wrap("points", alpha = 0.5, size = 0.8))) +
    scale_color_manual(values = c('red', 'goldenrod1'),
                       labels = c('Classe A', 'Classe B')) +
    scale_fill_manual(values = c('red', 'goldenrod1'),
                      labels = c('Classe A', 'Classe B')) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          legend.position = "right",
          legend.background = element_rect(fill = "white"))
)

# CORRELAZIONE
ggcorrplot(cor(saturn[,-9]), lab = T, outline.color = 'black') +
  theme_minimal() +
  labs(x = NULL, y = NULL) +
  labs(title = 'Correlation') +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"))


# BOXPLOT
saturn_scaled <- data.frame(cbind(scale(saturn[,-9]), saturn[,9]))
df1 <- saturn[,-9] %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value") %>%
  mutate(group = "Set 1")

df2 <- saturn_scaled[,-9] %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value") %>%
  mutate(group = "Set 2")

p1 <- ggplot(df1, aes(x = feature, y = value, fill = feature)) +
  geom_boxplot(outlier.shape = 8, outlier.size = 1) +
  scale_fill_manual(values = c('yellow','orange','red','green',4,5,6,8)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none", 
        plot.title = element_text(hjust = 0.5, size = 15, face = "bold")) +
  labs(title = "Boxplot Saturn", x = NULL, y = NULL)

p2 <- ggplot(df2, aes(x = feature, y = value, fill = feature)) +
  geom_boxplot(outlier.shape = 8, outlier.size = 1) +
  scale_fill_manual(values = c('yellow','orange','red','green',4,5,6,8)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none", 
        plot.title = element_text(hjust = 0.5, size = 15, face = "bold")) +
  labs(title = "Boxplot Scaled Saturn", x = NULL, y = NULL)

grid.arrange(p1, p2, ncol = 1) 


# HIST
par(mfrow = c(1, 1))

saturn_long <- saturn_scaled[, 1:8] %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value")

clrs <- c('yellow', 'orange', 'red', 'green', 4, 5, 6, 8)

ggplot(saturn_long, aes(x = value, fill = feature)) +
  geom_histogram(color = "black", bins = 20) +
  scale_fill_manual(values = clrs) +
  facet_wrap(~feature, nrow = 2) +
  theme_bw() +
  theme(legend.position = "none",
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    strip.text = element_text(face = "bold", size = 12)) +
  labs(title = "Histograms", x = NULL, y = "Density")


# TEST DI NORMALITA'
shapiro.test(saturn_scaled$feature.1)
shapiro.test(saturn_scaled$feature.2)
shapiro.test(saturn_scaled$feature.3)
shapiro.test(saturn_scaled$feature.4)
shapiro.test(saturn_scaled$feature.5)
shapiro.test(saturn_scaled$feature.6)
shapiro.test(saturn_scaled$feature.7)
shapiro.test(saturn_scaled$feature.8)

# GRAFICO SATURN -----------------------------------------------------------
library(plotly)

p <- plot_ly(
  saturn,
  x = ~feature.1,
  y = ~feature.2,
  z = ~feature.3,         
  color = ~factor(saturn$class),
  colors = c("red", "goldenrod2"),
  type = "scatter3d",
  mode = "markers",
  marker = list(size = 5)
) %>%
  layout(
    title = list(
      text = "<b>Class distribution</b><br><sup>Dataset Saturn</sup>",
      font = list(size = 22),
      x = 0.5, 
      y = 0.9,
      xanchor = "center"
    ),
    scene = list(
      xaxis = list(title = "Feature 1"),
      yaxis = list(title = "Feature 2"),
      zaxis = list(title = "Feature 3")
    ),
    legend = list(
      title = list(text = "Classe"),
      x = 0.8,
      y = 0.9,
      bordercolor = "black", 
      borderwidth = 1.5  
    )
  )

htmlwidgets::saveWidget(p, "grafico.html")

# GIF X PPT
dir.create("frames", showWarnings = FALSE)

p <- plot_ly(
  saturn,
  x = ~feature.1,
  y = ~feature.2,
  z = ~feature.3,         
  color = ~factor(saturn$class),
  colors = c("red", "goldenrod2"),
  type = "scatter3d",
  mode = "markers",
  marker = list(size = 5)
) %>%
  layout(
    title = list(
      text = "<b>Class distribution</b><br><sup>Dataset Saturn</sup>",
      font = list(size = 22),
      x = 0.5, 
      y = 0.9,
      xanchor = "center"
    ),
    scene = list(
      xaxis = list(title = "Feature 1"),
      yaxis = list(title = "Feature 2"),
      zaxis = list(title = "Feature 3"),
      camera = list(eye = list(x = 2, y = 2, z = 1))
    ),
    legend = list(
      title = list(text = "Classe"),
      x = 0.8,
      y = 0.9,
      bordercolor = "black", 
      borderwidth = 1.5  
    )
  )


dir.create("frames", showWarnings = FALSE)
angles <- seq(0, 360, by = 5)

for (i in angles) {
  p_temp <- p %>%
    layout(scene = list(
      camera = list(
        eye = list(
          x = 2 * cospi(i/180),
          y = 2 * sinpi(i/180),
          z = 1
        )
      )
    ))
  
  htmlwidgets::saveWidget(p_temp, "temp.html")
  webshot2::webshot("temp.html", file.path("frames", 
                                           sprintf("frame_%03d.png", i)),
                    vwidth = 1000, vheight = 600)
}

imgs <- list.files("frames", full.names = TRUE)
img_list <- lapply(imgs, image_read)
animation <- image_animate(image_join(img_list), fps = 10)
image_write(animation, "grafico_rotante.gif")


### K-MEANS ----------------------------------------------------------------
km.res <- kmeans(saturn[,-9], 2, iter.max = 100)
plot(saturn$feature.1, saturn$feature.2, col = km.res$cluster + 2) # NO
table(km.res$cluster, saturn$class)

# dati scalati
km_std.res <- kmeans(scale(saturn[,-9]), 2, iter.max = 100)
plot(saturn$feature.1, saturn$feature.2, col = km_std.res$cluster + 2) # NO
table(km_std.res$cluster, saturn$class)

# dati con minmax
saturn_mms <- apply(saturn, 2, function(x) (x-min(x))/(max(x)-min(x)))
km_std.res <- kmeans(saturn_mms[,-9], 2, iter.max = 100)
plot(saturn$feature.1, saturn$feature.2, col = km_std.res$cluster + 2) # NO
(mat1 <- table(km_std.res$cluster, saturn$class))
(acc1 <- sum(diag(mat1))/nrow(saturn))


### HIERARCHICAL CLUSTERING -----------------------------------------------
D <- dist(saturn_mms[,-9])
h.res <- hclust(D, method = "complete")
cluster <- cutree(h.res, k = 2)
plot(saturn$feature.1, saturn$feature.2, col = cluster + 2)
(mat2 <- table(cluster, saturn$class))
(acc2 <- sum(diag(mat2))/nrow(saturn))


### EM Clustering ---------------------------------------------------------
k <- 2
embj <- init.EM(saturn[,-9], nclass = k)
em.res <- emcluster(saturn[,-9], embj, assign.class = T)

ggplot(saturn, aes(x = feature.1, y = feature.2, 
                   color = factor(em.res$class))) +
  geom_point(size = 3) +
  scale_color_manual(values = c("red", "goldenrod2"),
                     labels = c("-1", "1")) +
  labs(color = "Class",
    x = "Feature 1",
    y = "Feature 2",
    title = "EM applicato a Saturn") +
  theme_minimal()


(mat3 <- table(em.res$class, saturn$class))
(acc3 <- sum(diag(mat3))/nrow(saturn))


### KNN ------------------------------------------------------------------
set.seed(123)

ixs <- createDataPartition(saturn$class, times = 1, p = 0.9)
dataset <- saturn[ixs$Resample1,]
testset <- saturn[-ixs$Resample1,]

# calcolo range solo sul training
featRanges <- apply(dataset[, -ncol(dataset)], 2, range)

for(j in 1:ncol(featRanges)) { 
  # applico standardizzazione al dataset e al testset con criterio max-min
  dataset[, j] <- (dataset[, j]-featRanges[1, j])/diff(featRanges[, j])
  testset[, j] <- (testset[, j]-featRanges[1, j])/diff(featRanges[, j])
}


# K fold CV per la scelta del k (# neighbours) ottimale 
knn_k  <- 10:40

set.seed(123)
ixs <- createFolds(dataset$class, k = 10, list = T)

trnErrMeans <- valErrMeans <- numeric(length(knn_k))

for(i in 1:length(knn_k)) {
  trnErrs <- valErrs <- numeric()
  for(k in 1:length(ixs)) {
    
    valFold <- dataset[ixs[[k]], ]
    trnFold <- dataset[-ixs[[k]], ]
    
    preds <- knn3Train(train = trnFold[, -ncol(trnFold)],
                       test = trnFold[, -ncol(trnFold)],
                       cl = trnFold$class, k = knn_k[i], prob = F, use.all = T)
    trnErrs[k] <- mean(trnFold$class != preds) 
    
    preds <- knn3Train(train = trnFold[, -ncol(trnFold)],
                       test = valFold[, -ncol(valFold)],
                       cl = trnFold$class, k = knn_k[i], prob = F, use.all = T)
    valErrs[k] <- mean(valFold$class != preds) 
  }
  
  trnErrMeans[i] <- mean(trnErrs)
  valErrMeans[i] <- mean(valErrs)
  
}


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
preds.fin <- knn3Train(train = dataset[, -ncol(dataset)], 
                       test = testset[, -ncol(dataset)], cl = dataset$class,
                       k = 20, prob = F, use.all = T)
(predsErr <- mean(as.factor(preds.fin) != testset$class))
(1 - predsErr)*100


# GRAFICO GGPLOT KNN GENERALIZATION VS EMPIRICAL -----------------------------
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
    legend.text = element_text(size = 15),      
    legend.title = element_text(size = 15),
    legend.background = element_rect(fill = "white")) 



### SVM --------------------------------------------------------------------
mdl <- svm(x = saturn[,1:(ncol(saturn)-1)], y = saturn[,ncol(saturn)],
           type = "C-classification", kernel = "radial", scale = T)
preds <- predict(mdl, saturn[,1:(ncol(saturn)-1)])
(confM <- table(preds, saturn[,ncol(saturn)]))

plot(saturn$feature.1, saturn$feature.2, col = as.numeric(preds) + 2, pch = 19)
(acc4 <- sum(diag(confM))/nrow(saturn))

# BO -----------------------------------------------------------------------
set.seed(123)
par.set <- makeParamSet(makeNumericParam("cost", lower = -2, upper = 1.5, 
                                         trafo = function(x) 10^x),
                        makeNumericParam("gamma", lower = -4, upper = 1, 
                                         trafo = function(x) 10^x))
ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 15)
tune.ctrl <- makeTuneControlMBO(mbo.control = ctrl)

saturn$class <- as.factor(saturn$class)
task <- makeClassifTask(data = saturn, target = "class")
cv3 <- makeResampleDesc("CV", iters = 5)
run <- tuneParams(makeLearner("classif.svm"), task, cv3, measures = acc, 
                  par.set = par.set, control = tune.ctrl)
ys <- getOptPathY(run$opt.path)
plot(ys, type = "o", pch = 19, lwd = 3, lty = 2, col = "grey")
lines(cummax(ys), type = "s", lwd = 3, col = "green")

# GRID SEARCH ---------------------------------------------------------------
set.seed(349)

folds <- createFolds(y = saturn[,ncol(saturn)], k = 10, returnTrain = T, 
                     list = T)
C_values <- c(0.01, 1, 100, 1000, 10000)
g_values <- c(0.0001, 0.001, 0.1, 1, 10)


results <- data.frame(C = numeric(), gamma = numeric(),
                      empErr = numeric(), genErr = numeric(), svPerc = numeric())


# rieseguo il modello con parametri ottimali secondo BO
run$x 
mdl.opt <- svm(x = saturn[,1:(ncol(saturn)-1)], y = saturn[,ncol(saturn)],
               type = "C-classification", kernel = "radial", scale = T,
               cost = run$x$cost, gamma = run$x$gamma)
preds <- predict(mdl.opt, saturn[,1:(ncol(saturn)-1)])
(confM <- table(preds, saturn[, ncol(saturn)]))

plot(saturn$feature.1, saturn$feature.2, col = preds)
cat("Empirical error:", round(100*(1-(sum(diag(confM))/sum(confM))), 2),"%\n")
cat("number of svs:", mdl.opt$nSV, "(",
    round(100*mdl.opt$tot.nSV/nrow(saturn), 2),"%)\n")
(acc4 <- sum(diag(confM))/nrow(saturn))


# per ogni c e per ogni gamma rifaccio 10 fold cv
for(c in C_values){
  for(g in g_values){
    
    empErr <- genErr <- percSVs <- numeric()
    for(i in 1:length(folds)){
      
      # divide data e labels
      trn_x <- saturn[folds[[i]], -ncol(saturn)]
      trn_y <- saturn[folds[[i]], ncol(saturn)]
      
      # facciamo la stessa cosa anche con il validation
      val_x <- saturn[-folds[[i]], -ncol(saturn)]
      val_y <- saturn[-folds[[i]], ncol(saturn)]
      
      mdl <- svm(x = trn_x, y = trn_y, type = "C-classification",
                 kernel = "radial", scale = F, cost = c, gamma = g)
      # c = cost costo di generalizzazione 
      
      preds <- predict(mdl, trn_x)
      empErr[i] <- mean(preds != trn_y)
      
      percSVs[i] <- mdl$tot.nSV / length(folds[[i]])
      # percentuale di punti che stiamo usando come support vectors
      # solo nel training !!!
      
      preds <- predict(mdl, val_x)
      genErr[i] <- mean(preds != val_y)
      
    }
    
    results <- rbind(results, data.frame(C = c, gamma = g,
                                         empErr = mean(empErr),
                                         genErr = mean(genErr),
                                         svPerc = mean(percSVs)))
  }
}

ys <- 1 - results$genErr  # accuratezza media di validazione

# GRAFICO BEST SEEN BO VS GRID SEARCH --------------------------------------
ys_grid <- 1 - results$genErr
df_grid <- data.frame(
  Iter = 1:length(ys_grid),
  Accuracy = ys_grid,
  Cummax = cummax(ys_grid),
  Method = "Grid Search"
)

ys_bo <- getOptPathY(run$opt.path)
df_bo <- data.frame(
  Iter = 1:length(ys_bo),
  Accuracy = ys_bo,
  Cummax = cummax(ys_bo),
  Method = "Bayesian Optimization"
)

df_all <- bind_rows(df_grid, df_bo)

ggplot(df_all, aes(x = Iter, color = Method, group = Method)) +
  # linea tratteggiata per Accuracy media
  geom_line(aes(y = Accuracy*100), linetype = "dotted", size = 1, alpha = 0.6) +
  geom_point(aes(y = Accuracy*100), size = 2) +
  # linea gradino per Best Seen
  geom_step(aes(y = Cummax*100), linetype = "solid", size = 1.2) +
  scale_color_manual(values = c("Grid Search" = "red", 
                                "Bayesian Optimization" = "green")) +
  labs(
    title = "Grid Search vs Bayesian Optimization",
    x = "Iteration",
    y = "Accuracy (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_blank()
  )

# BO CON DIVISIONE TRA TRAIN E TEST ----------------------------------------- 
set.seed(123)
n <- nrow(saturn)

train.idx <- sample(1:n, 0.9*n)
train <- saturn[train.idx, ]
test  <- saturn[-train.idx, ]

task.train <- makeClassifTask(data = train, target = names(saturn)[ncol(saturn)])

cv3 <- makeResampleDesc("CV", iters = 5)

par.set <- makeParamSet(
  makeNumericParam("cost",  lower = -2, upper = 1.5, trafo = function(x) 10^x),
  makeNumericParam("gamma", lower = -4, upper = 1, trafo = function(x) 10^x))
ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 15)
tune.ctrl <- makeTuneControlMBO(mbo.control = ctrl)

# misura accuratezza + percentuale svs
svm_complexity_measure <- makeMeasure(
  id = "svm_simple",
  name = "SVM Simplicity Score",
  properties = c("classif", "req.model"), 
  minimize = TRUE,
  best = 0,
  worst = Inf,
  fun = function(task, model, pred, feats, extra.args) {
    accuracy <- as.numeric(performance(pred, measures = mlr::acc))
    n_sv <- model$learner.model$tot.nSV
    n_total <- nrow(getTaskData(task))
    perc_sv <- n_sv / n_total
    lambda <- 0.5
    out <- (1 - accuracy) + lambda * perc_sv
    print(list(acc = accuracy, n_sv = n_sv, n_total = n_total, score = out))
    return(out)
  }
)

run <- tuneParams(makeLearner("classif.svm"), task = task.train, 
                  resampling = cv3,
                  measures = list(svm_complexity_measure), 
                  par.set = par.set, control = tune.ctrl)
run$x
mdl <- svm(x = train[, -ncol(train)], y = train[, ncol(train)], 
           type = "C-classification",
           kernel = "radial", scale = T, cost = run$x$cost, gamma = run$x$gamma)
preds.test <- predict(mdl, test[, -ncol(test)])
(confM.test <- table(preds.test, test[, ncol(test)]))
cat("Error:",
    round(100*(1-(sum(diag(confM.test))/sum(confM.test))), 2),"%\n")
cat("number of svs:", mdl$nSV,"(", round(100*mdl$tot.nSV/nrow(train), 2),"%)\n")
(acc.test <- sum(diag(confM.test))/sum(confM.test))

# applico su tutto il dataset
mdl <- svm(x = saturn[, -ncol(saturn)], y = saturn[, ncol(saturn)], 
           type = "C-classification",
           kernel = "radial", scale = T, cost = run$x$cost, gamma = run$x$gamma)
cat("Error:",
    round(100*(1-(sum(diag(confM.test))/sum(confM.test))), 2),"%\n")
cat("number of svs:", mdl$nSV,"(", round(100*mdl$tot.nSV/nrow(saturn), 2),"%)\n")
(acc.test <- sum(diag(confM.test))/sum(confM.test))

# GRAFICO 3D DEL CONFINE DECISIONALE SVM -----------------------------------
svm_model <- svm(
  class ~ feature.1 + feature.2 + feature.3,
  data = saturn,
  kernel = "radial",
  decision.values = TRUE
)

fixed_z <- mean(saturn$feature.3)

x_seq <- seq(min(saturn$feature.1), max(saturn$feature.1), length.out = 80)
y_seq <- seq(min(saturn$feature.2), max(saturn$feature.2), length.out = 80)
grid <- expand.grid(feature.1 = x_seq, feature.2 = y_seq, feature.3 = fixed_z)

grid$decision <- attributes(predict(svm_model, grid, decision.values = TRUE))$decision.values

z_matrix <- matrix(grid$decision, nrow = length(x_seq), ncol = length(y_seq))

plot_ly() %>%
  add_markers(
    data = saturn,
    x = ~feature.1, y = ~feature.2, z = ~feature.3,
    color = ~factor(class),
    colors = c("red", "goldenrod2"),
    marker = list(size = 4)
  ) %>%
  add_surface(
    x = x_seq, y = y_seq, z = z_matrix,
    showscale = FALSE,
    opacity = 0.6,
    surfacecolor = z_matrix,
    colorscale = list(c(0, 'lightgray'), c(1, 'lightgray'))
  ) %>%
  layout(
    title = list(text = "<b>SVM RBF Decision Boundary (slice)</b>"),
    scene = list(
      xaxis = list(title = "Feature 1"),
      yaxis = list(title = "Feature 2"),
      zaxis = list(title = "Feature 3")
    )
  )


### RANDOM FOREST ----------------------------------------------------------
set.seed(12345)
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
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.text = element_text(size = 15),      
        legend.title = element_text(size = 15),
        legend.position = c(0.8, 0.8),
        legend.background = element_rect(fill = "white"))
