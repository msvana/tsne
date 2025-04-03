type TSNEConfig = {
    nDims?: number;
    perplexity?: number;
    learningRate?: number;
    nIter?: number;
};

const defaultConfig: TSNEConfig = {
    nDims: 2,
    perplexity: 30,
    learningRate: 1.0,
    nIter: 100,
};

export class DataValidationError extends Error {}

export class TSNE {
    #config: TSNEConfig;
    #n: number;

    #sigmas: number[];
    #affinities: number[][];
    #sigmasMaxIterations: number = 50;

    #distances: number[];
    #distancesY: number[];
    #invertedDistancesY: number[];
    #projectedAffinities: number[];

    #gradient: number[];

    constructor(config: TSNEConfig = {}) {
        this.#config = { ...defaultConfig, ...config };
    }

    transform(X: number[][]): number[][] {
        this.#validateInput(X);
        this.#n = X.length;

        this.#distances = new Array(this.#n * this.#n).fill(0);
        this.#getPairwiseDistances(X, this.#distances);
        this.#sigmas = this.#findBestSigmas();

        this.#affinities = this.#getSymmetricAffinities();
        this.#distancesY = new Array(this.#n * this.#n).fill(0);
        this.#invertedDistancesY = new Array(this.#n * this.#n).fill(0);
        this.#projectedAffinities = new Array(this.#n * this.#n).fill(0);
        this.#gradient = new Array(this.#n * this.#config.nDims);

        let momentum: number;

        let Y = this.#initRandomProjection();
        let YPrev = structuredClone(Y);
        let YNew = structuredClone(Y);

        for (let i = 0; i < this.#config.nIter; i++) {
            momentum = this.#getMomentum(i);
            this.#updateGradient(Y);

            for (let j = 0; j < this.#n; j++) {
                for (let k = 0; k < this.#config.nDims; k++) {
                    YNew[j][k] = Y[j][k];
                    YNew[j][k] -=
                        this.#gradient[j * this.#config.nDims + k] * this.#config.learningRate;
                    YNew[j][k] += (Y[j][k] - YPrev[j][k]) * momentum;

                    YPrev[j][k] = Y[j][k];
                    Y[j][k] = YNew[j][k];
                }
            }
        }

        return Y;
    }

    #validateInput(X: number[][]) {
        if (!Array.isArray(X)) {
            throw new DataValidationError(`Input must be an array`);
        }

        if (X.length < 2) {
            throw new DataValidationError(
                `Input must have at least 2 rows (vectors). Current input has ${X.length} rows`,
            );
        }

        for (const i in X) {
            if (!Array.isArray(X[i])) {
                throw new DataValidationError(
                    `Elements of the input array must be arrays. Row ${i} is not an array`,
                );
            }

            if (X[i].length !== X[0].length) {
                throw new DataValidationError(
                    `Rows must have the same length. Row 0 has ${X[0].length} items, but row ${i} has ${X[i].length} items.`,
                );
            }

            if (!X[i].every((item) => typeof item === "number")) {
                throw new DataValidationError(
                    `All items in a a row must be numbers. Row ${i} contains items that are not numbers.`,
                );
            }
        }
    }

    #getPairwiseDistances(X: number[][], target: number[]) {
        const n = X.length;
        let distance: number;

        for (let i = 0; i < n; i++) {
            for (let j = i; j < n; j++) {
                if (i === j) {
                    target[i * n + j] = 0;
                    continue;
                }

                distance = squaredEuclideanDistance(X[i], X[j]);
                target[i * n + j] = distance;
                target[j * n + i] = distance;
            }
        }
    }

    #findBestSigmas(): number[] {
        const initialLowerBound = 1e-3;
        const initialUpperBound = this.#getInitialSigmaUpperBound();
        const sigmas: number[] = new Array(this.#n);
        const affinities: number[] = new Array(this.#n);

        let lowerBound: number;
        let upperBound: number;
        let sigma: number;
        let pointDistances: number[];
        let perplexity: number;

        for (let i = 0; i < this.#n; i++) {
            lowerBound = initialLowerBound;
            upperBound = initialUpperBound;
            pointDistances = this.#distances.slice(i * this.#n, (i + 1) * this.#n);

            for (let j = 0; j < this.#sigmasMaxIterations; j++) {
                sigma = (upperBound + lowerBound) / 2;
                this.#getAffinities(pointDistances, sigma, i, affinities);
                perplexity = this.#getPerplexity(affinities);

                if (Math.abs(perplexity - this.#config.perplexity) <= 1e-3) {
                    break;
                }

                if (perplexity < this.#config.perplexity) {
                    lowerBound = sigma;
                } else {
                    upperBound = sigma;
                }
            }

            sigmas[i] = sigma;
        }

        return Array.from(sigmas);
    }

    #getSymmetricAffinities(): number[][] {
        const affinities = new Array(this.#n);
        const symmetricAffinities = new Array(this.#n);
        let pointDistances: number[];

        for (let i = 0; i < this.#n; i++) {
            pointDistances = this.#distances.slice(i * this.#n, (i + 1) * this.#n);
            affinities[i] = new Array(this.#n);
            this.#getAffinities(pointDistances, this.#sigmas[i], i, affinities[i]);
            symmetricAffinities[i] = new Array(this.#n).fill(0);
        }

        let p: number;

        for (let i = 0; i < this.#n; i++) {
            for (let j = i + 1; j < this.#n; j++) {
                p = (affinities[i][j] + affinities[j][i]) / (2 * this.#n);
                symmetricAffinities[i][j] = p;
                symmetricAffinities[j][i] = p;
            }
        }

        return symmetricAffinities;
    }

    #initRandomProjection(): number[][] {
        const Y: number[][] = new Array(this.#n);

        for (let i = 0; i < this.#n; i++) {
            Y[i] = new Array(this.#config.nDims);

            for (let j = 0; j < this.#config.nDims; j++) {
                Y[i][j] = Math.random() - 0.5;
            }
        }

        return Y;
    }

    #getMomentum(iter: number): number {
        const momentumStart = 0.5;
        const momentumEnd = 0.8;

        const momentum =
            momentumStart + ((momentumEnd - momentumStart) * iter) / this.#config.nIter;
        return momentum;
    }

    #updateGradient(Y: number[][]) {
        this.#getPairwiseDistances(Y, this.#distancesY);
        invertedDistances(this.#distancesY, this.#n, this.#invertedDistancesY);
        this.#updateProjectedAffinities();
        this.#gradient.fill(0);

        let coef: number;
        let idx: number;

        for (let i = 0; i < this.#n; i++) {
            for (let j = 0; j < this.#n; j++) {
                idx = i * this.#n + j;
                coef = this.#affinities[i][j] - this.#projectedAffinities[idx];
                coef *= this.#invertedDistancesY[idx];

                for (let k = 0; k < this.#config.nDims; k++) {
                    this.#gradient[i * this.#config.nDims + k] += 4 * coef * (Y[i][k] - Y[j][k]);
                }
            }
        }
    }

    #getInitialSigmaUpperBound(): number {
        const maxDistance = this.#distances.reduce((max, curr) => (curr > max ? curr : max), 0);
        const upperBound = Math.log(maxDistance + 1e-6) + 10;
        return upperBound;
    }

    #getAffinities(
        pointDistances: number[],
        sigma: number,
        rowNo: number,
        affinitiesOut: number[],
    ) {
        let sum = 0;
        const denominator = 2 * sigma * sigma;

        for (let i = 0; i < this.#n; i++) {
            if (i === rowNo) {
                affinitiesOut[i] = 0;
                continue;
            }

            affinitiesOut[i] = Math.exp(-pointDistances[i] / denominator);
            sum += affinitiesOut[i];
        }

        for (let i = 0; i < this.#n; i++) {
            affinitiesOut[i] = affinitiesOut[i] / sum;
        }
    }

    #getPerplexity(affinities: number[]): number {
        let H = 0;

        for (let i = 0; i < affinities.length; i++) {
            H -= affinities[i] * Math.log2(affinities[i] + 1e-10);
        }

        return Math.pow(2, H);
    }

    #updateProjectedAffinities() {
        let sum = 0;

        for (let i = 0; i < this.#projectedAffinities.length; i++) {
            sum += this.#invertedDistancesY[i];
        }

        for (let i = 0; i < this.#projectedAffinities.length; i++) {
            this.#projectedAffinities[i] = this.#invertedDistancesY[i] / sum;
        }
    }
}

export function squaredEuclideanDistance(a: number[], b: number[]): number {
    let sum: number = 0;
    let dimDifference: number;

    for (let i = 0; i < a.length; i++) {
        dimDifference = a[i] - b[i];
        sum += dimDifference * dimDifference;
    }

    return sum;
}

export function invertedDistances(distances: number[], n: number, target: number[]) {
    let d: number;

    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            d = 1 / (1 + distances[i * n + j]);
            target[i * n + j] = d;
            target[j * n + i] = d;
        }
    }
}

function printMatrix(matrix: number[], nRows: number) {
    const nCols = matrix.length / nRows;

    console.log("---");

    for (let i = 0; i < nRows; i++) {
        const row = matrix.slice(i * nCols, (i + 1) * nCols);
        let rowOutput = "";

        for (let j = 0; j < nCols; j++) {
            rowOutput += `${row[j].toFixed(3)} `;
        }

        console.log(rowOutput);
    }
}
