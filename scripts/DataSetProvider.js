DATA_SET_05 = getDataSet("DataSet5.csv");
DATA_SET_10 = getDataSet("DataSet10.csv");

let ds05 = [];

async function getDataSet(file) {
    const ds = tf.data.csv(LOCAL_DATA + file, {
            delimiter: ";"
        }
    );

    const dataArray = await ds.toArray().then(result => {
        console.log(result);
        return result;
    });
}




