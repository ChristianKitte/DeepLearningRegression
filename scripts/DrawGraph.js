//////////////////////////////////////////////////////////////////////////////////////////
////
//// Hält alle Funktionalität zur Ausgabe von Daten in en Plotly Diagramm
////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * Gibt ein Plotly Scatter Plot innerhalb des übergebenen Containers aus. Hierfür werden alle auszugebenden
 * Daten als Array übergeben.
 * @param divID Die ID eines DIV Elementes als Container
 * @param title Der Titel der Ausgabe
 * @param dataArray Die Labels und Features als Array
 * @constructor
 */
function DrawGraph(divID, title, dataArray) {
    let featureArray = [];
    let labelArray = [];

    for (let element of dataArray) {
        featureArray.push(element.feature);
        labelArray.push(element.label);
    }

    var valuesDefinition = {
        x: featureArray,
        y: labelArray,
        mode: "markers",
        type: 'scatter',
        name: 'Funktionswerte'
    };

    var data = [valuesDefinition];

    var layout = {
        title: title,
        width: 500,
        height: 500,
        xaxis: {
            title: 'Eingabe (X-Wert)',
            range: [-1, 1],
            showgrid: false,
            zeroline: false
        },
        yaxis: {
            title: 'Funktionswert (Y-Wert)',
            range: [-1, 1]
        }
    };

    Plotly.newPlot(divID, data, layout);
}