

plotResults <- function(threshold, bridgeRatio, errorRatio) {
  par(mfrow=c(1,3))
  
  plot(threshold, errorRatio, type = "l", main = "Poziom błędu (%)", xlab = "Próg pewności", ylab = "Poziom błędu (%)", col="red", col.axis = "dimgray", col.lab = "blueviolet")
  plot(threshold, bridgeRatio, type = "l", main = "Poziom klasyfikacji (%)", xlab = "Próg pewności", ylab = "Poziom klasyfikacji (%)", col="red", col.axis = "dimgray", col.lab = "blueviolet")
  plot(bridgeRatio, errorRatio, type = "l", main = "Poziom klasyfikacji vs. Poziom błędu", xlab = "Poziom klasyfikacji (%)", ylab = "Poziom błędu (%)", col="red", col.axis = "dimgray", col.lab = "blueviolet")
  
}