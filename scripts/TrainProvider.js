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

// Helper functions used for sampling
function getUniformDistributedRandomNumber(min, max) {
    return Math.random() * (max - min) + min;
}

function testModel(trainedModel) {
    let Testdaten = [];

    for (let i = 0; i < 100; i++) {
        let x = getUniformDistributedRandomNumber(-1, 1);
        let y = (x + 0.8) * (x - 0.2) * (x - 0.3) * (x - 0.6);

        Testdaten.push({
            features: x,
            labels: y,
        });
    }

    const currentData = getData();

    const predictions = trainedModel.predict(normalizedTestFeatures);

    const predictedPoints = Array.from(Testdaten).map((val, i) => {
        return {x: val.features, y: unNormalizedPredictions[i]}
    });

    const originalPoints = Array.from(Testdaten).map((val, i) => {
        return {x: val.features, y: val.labels}
    });

    tfvis.render.scatterplot(
        {name: "Model Predictions vs Original Data"},
        {values: [originalPoints, predictedPoints], series: ["original", "predicted"],},
        {
            xLabel: "x",
            yLabel: "y",
            height: 300,
        }
    );
}