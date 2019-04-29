


s <- seq(0,2,0.1)

plot(s, exp(s), type="l")


/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_ensemble_5-30_topics_20_iters_each_2



rootFrom='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_ensemble_5-30_topics_20_iters_each_2/all'

for(j in 9:19){
  for(i in 0:4){
    base = paste(rootFrom,'/',j,sep='')
    file.rename(from = file.path(base, paste('',i, sep='')), to = file.path(base, paste('',i+j*5, sep='')))
  }
}


citation(package = "lda", lib.loc = NULL)
