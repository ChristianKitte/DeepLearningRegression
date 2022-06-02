async function run() {
    if (isArrayAndFilled(currentDataSet)) {
        let shuffled = getShuffledFeatureAndLabelColumn(currentDataSet);

        let featureTensor = tf.tensor(shuffled.features);
        let labelTensor = tf.tensor(shuffled.labels);

        featureTensor.print();
        labelTensor.print();

        let normalizedfeatures = normalizeTensor(featureTensor);
        let normalizedLabels = normalizeTensor(labelTensor);

        normalizedfeatures.print();
        normalizedLabels.print();

        const model = createModel();

        tfvis.show.modelSummary(document.getElementById("a"), model);

        trainModel(model, normalizedfeatures, normalizedLabels).then(
            () => {
                const series1 = Array(100).fill(0)
                    .map(y => Math.random() * 100 - (Math.random() * 50))
                    .map((y, x) => ({x, y,}));

                const series2 = Array(100).fill(0)
                    .map(y => Math.random() * 100 - (Math.random() * 150))
                    .map((y, x) => ({x, y,}));

                const series = ['First', 'Second'];
                const data = {values: [series1, series2], series}

                //const surface = {name: 'Scatterplot', tab: 'Charts'};
                //tfvis.render.scatterplot(surface, data);

                tfvis.render.scatterplot(document.getElementById("c"), data);

                //testModel(model, featureTensor, normalizedLabels);
            }
        );
    } else {
        alert("Bitte wählen Sie ein DataSet aus!");
    }
}

function createModel() {
    // Ein sequentielles, LAyer basiertes Modell
    let model = tf.sequential();

    model.add(
        tf.layers.dense({
            inputShape: [1],
            units: 1,
            activation: "sigmoid"
        })
    );
    model.add(
        tf.layers.dense({
            inputShape: [1],
            units: 1,
            activation: "sigmoid"
        })
    );
    model.add(tf.layers.dense({units: 1}));

    return model;
}

async function trainModel(model, inputs, labels) {
    model.compile({
        optimizer: tf.train.sgd(0.001),
        loss: "meanSquaredError",
        metrics: [tf.metrics.meanAbsoluteError]
    });

    let batchSize = 32;
    let epochs = 100;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            document.getElementById("b"),
            ['loss', 'mse'],
            {height: 200, callbacks: ['onEpochEnd']}
        )
    });
}

function testModel(model, inputs, labels) {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const predictedPoints = {
        x: xs,
        y: preds[1]
    };

    tfvis.render.scatterplot(
        document.getElementById("c"),
        {values: [predictedPoints], series: ['predicted']},
        {
            xLabel: 'X-Wert',
            yLabel: 'Y-Wert',
            height: 300
        }
    );
}