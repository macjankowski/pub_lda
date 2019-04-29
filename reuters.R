
library("tm")
reut21578XMLgz <- system.file("texts", "/Users/mjankowski/doc/data/reut21578.xml.gz", package = "tm")

TextDocCol(ReutersSource(gzfile(reut21578XMLgz)), readerControl = list(reader = readReut21578XML, language = "en_US", load = FALSE))
TextDocCol
