//////////////////////////////////////////////////////////////////////////////////////////
////
//// Hält alle Funktionalität Im Zusammenhang mit der Erzeugung von Zufallszahlen und dem
//// Überlagern mit einem Rauschen
////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * Erzeugt eine zufällige, gleichberteilte Nummer im Bereich min bis max
 * @param min Die untere Grenze
 * @param max Die obere Grenze
 * @returns {*} Die Zufallszahl
 */
function getUniformDistributedRandomNumber(min, max) {
    return Math.random() * (max - min) + min;
}

/**
 * Fügt dem übergebenen Wert ein  Gausches Rauschen, basierend auf einen Mittelwert und Varianz hinzu
 * @param label Der zu überlagernde Wert
 * @param mean Der zu verwendende Mittelwert
 * @param variance Die zu verwendende Varianz
 * @returns {*} Der überlagerte Wert
 */
function addGaussianNoise(label, mean, variance) {
    return label * Math.sqrt(variance) + mean;
}

/**
 * Berechnet die Summe des übergebenen, zweidimensionalen Arrays von Zahlen
 * @param dataArray Das Array, über das die Summe berechnet werden soll
 * @returns {number} Die berechnete Summe des Arrays
 */
function calcSum(dataArray) {
    let total = 0;

    for (let i = 0; i < dataArray.length; i++) {
        total += dataArray[i].label;
    }

    return total;
}

/**
 * Berechnet den Mittelwert des übergebenen, zweidimensionalen Arrays von Zahlen
 * @param dataArray Das Array, über das der Mittelwert berechnet werden soll
 * @returns {number} Der berechnete Mittelwert des Arrays
 */
function calcMean(dataArray) {
    const sum = calcSum(dataArray);
    return sum / dataArray.length;
}

/**
 * Überlagert ein übergebenes Array mit einem Rauschen in vorgegebener Varianz
 * @param dataArray Das zu bearbeitende Array
 * @param varianz Die anzuwendende Varianz
 * @returns {*} Das mit einem Rauschen überlagerte Array
 */
function getDataArrayWithNoise(dataArray, varianz) {
    const mean = calcMean(dataArray);

    for (let item of dataArray) {
        let newLabel = addGaussianNoise(item.label, mean, varianz);
        item.label = newLabel;
    }

    return dataArray;
}

/**
 * Erzeugt ein Array mit der angegebenen Anzahl an gleichverteilten, zufälligen Zahlen als feature im angegebenen
 * Bereich. Das label ergibt sich aus der Funktion: (feature + 0.8) * (feature - 0.2) * (feature - 0.3) * (feature - 0.6)
 * @param countNumbers Die Anzahl der zu erzeugenden Zahlen
 * @param min Die untere Grenze
 * @param max Die obere Grenze
 * @returns {*[]}
 */
function getUniformRandomDataArray(countNumbers, min, max) {
    let dataArray = [];

    for (let i = 0; i < countNumbers; i++) {
        let feature = getUniformDistributedRandomNumber(min, max);
        let label = (feature + 0.8) * (feature - 0.2) * (feature - 0.3) * (feature - 0.6);
        //let label = feature + 2 * feature;

        dataArray.push({
            feature: feature,
            label: label,
        });
    }

    return dataArray;
}