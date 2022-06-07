/**
 * Trainiert das übergebene, kompilierte Modell aus Basis der übergebenen Features und Labels.
 * @param compiledModel Das kompilierte Model
 * @param inputs Die für das Training vorgesehen Features
 * @param labels Die zugehörigen Labes
 * @param batchSize Die Größe eines einzelnen Batches je Epoche
 * @param epochs Die Anzahl der Epochen
 * @param divIdVisuelleAusgabe Die ID eine DIV Containers zur Ausgabe der Metriken. Ausgegeben werden fest 'loss'
 * und 'mse'
 */
async function trainModel(compiledModel, inputs, labels, batchSize, epochs, divIdVisuelleAusgabe) {
    return await compiledModel.fit(
        inputs,
        labels, {
            batchSize: batchSize,
            epochs: epochs,
            shuffle: true,
            validationSplit: 0.1,
            callbacks: tfvis.show.fitCallbacks(
                document.getElementById(divIdVisuelleAusgabe),
                ['loss', 'mse'],
                {
                    height: 200,
                    callbacks: ['onEpochEnd']
                }
            )
        }
    );
}

/**
 * Tested das übergebene und trainierte Modell auf Basis der dieser Anwendung zugrunde liegenden Funktion als
 * GroundTruth.
 * @param trainedModel
 * @param divIdVisuelleAusgabe
 */
function testModel(trainedModel, divIdVisuelleAusgabe) {
    // Ganz neue Testdaten vorbereiten und normalisieren. Feature und Label liegen normalisiert vor
    //let Testdaten = getDataArray(100);
    //testDataSet = getData(Testdaten);

    //const {maxFeature, minFeature, minLabel, maxLabel} = currentDataSet;

    const [testFeatures, testPredicts] = tf.tidy(() => {
        const testFeatures = tf.linspace(-1, 1, 100);
        let x = normalizeTensor(testFeatures.reshape([100, 1]), currentTrainData.maxFeature, currentTrainData.minFeature,)
        const testPredicts = trainedModel.predict(x.normierterTensor);
        let y = deNormalizeTensor(testPredicts, currentTrainData.minLabel, currentTrainData.maxLabel)
        //const unNormXs = testFeatures.mul(maxFeature.sub(minFeature)).add(minFeature);
        //const unNormPreds = testPredicts.mul(maxLabel.sub(minLabel)).add(minLabel);

        // Un-normalize the data
        return [testFeatures.dataSync(), y.dataSync()];
    });

    // PunkteArray für die nicht normalisierten Originalpunkte erstellen (GroundTruth)
    let originalPoints = [];
    let Testdaten = getFullDataArray(-1, 1, 0.01);
    Testdaten.forEach(function (value, i) {
        originalPoints.push(
            {x: value.feature, y: value.label}
        );
    });

    // PunkteArray für die vorhergesagten Punkte erstellen (GroundTruth)
    let predictedPoints = [];
    for (let i = 0; i < 100; i++) {
        predictedPoints.push(
            {x: testFeatures[i], y: testPredicts[i]}
        );
    }

    tfvis.render.scatterplot(
        document.getElementById(divIdVisuelleAusgabe),
        {values: [originalPoints, predictedPoints], series: ["original", "predicted"],},
        {
            xLabel: "x",
            yLabel: "y",
            height: 300,
        }
    );
}