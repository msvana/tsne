import * as mnist from "mnist";
import { TSNE } from "./index";

const dataset = mnist.default.set(250, 0).training;
const inputs = dataset.map((item: any) => item.input);
const classes = dataset.map((item: any) => item.output.indexOf(1));

const tsne = new TSNE({ nIter: 200, perplexity: 30, learningRate: 10 });

const projections = tsne.transform(inputs);
const chartContainer = document.getElementById("chart");

const data = projections.map((p) => {
    return { x: p[0], y: p[1] };
});

const colorMap = {
    0: "red",
    1: "green",
    2: "blue",
    3: "black",
    4: "cyan",
    5: "magenta",
    6: "yellow",
    7: "orange",
    8: "brown",
    9: "gray",
};

const colors = classes.map((cls: number) => colorMap[cls]);

new Chart(chartContainer, {
    type: "scatter",
    data: {
        datasets: [
            {
                data: data,
                backgroundColor: colors,
            },
        ],
    },
});
