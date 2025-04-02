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
    #distances: number[];
    #sigmas: number[];
    #affinities: number[][];
    #sigmasMaxIterations: number = 50;

    constructor(config: TSNEConfig = {}) {
        this.#config = { ...defaultConfig, ...config };
    }

    transform(X: number[][]): number[][] {
        this.#validateInput(X);
        this.#n = X.length;
        this.#distances = this.#getPairwiseDistances(X);
        this.#sigmas = this.#findBestSigmas();
        this.#affinities = this.#getSymmetricAffinities();

        let momentum: number;
        let gradient: number[];

        let Y = this.#initRandomProjection();
        let YPrev = structuredClone(Y);
        let YNew: number[][];

        for (let i = 0; i < this.#config.nIter; i++) {
            momentum = this.#getMomentum(i);
            gradient = this.#getGradient(Y);
            YNew = new Array(this.#n);

            for (let j = 0; j < this.#n; j++) {
                YNew[j] = new Array(this.#config.nDims);

                for (let k = 0; k < this.#config.nDims; k++) {
                    YNew[j][k] = Y[j][k];
                    YNew[j][k] -= gradient[j * this.#config.nDims + k] * this.#config.learningRate;
                    YNew[j][k] += (Y[j][k] - YPrev[j][k]) * momentum;
                }
            }

            YPrev = Y;
            Y = YNew;
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

    #getPairwiseDistances(X: number[][]): number[] {
        const n = X.length;
        const distances = new Array(n * n);

        let distance: number;

        for (let i = 0; i < n; i++) {
            for (let j = i; j < n; j++) {
                if (i === j) {
                    distances[i * n + j] = 0;
                    continue;
                }

                distance = squaredEuclideanDistance(X[i], X[j]);
                distances[i * n + j] = distance;
                distances[j * n + i] = distance;
            }
        }

        return distances;
    }

    #findBestSigmas(): number[] {
        const initialLowerBound = 1e-3;
        const initialUpperBound = this.#getInitialSigmaUpperBound();
        const sigmas: number[] = new Array(this.#n);

        let lowerBound: number;
        let upperBound: number;
        let sigma: number;
        let pointDistances: number[];
        let affinities: number[];
        let perplexity: number;

        for (let i = 0; i < this.#n; i++) {
            lowerBound = initialLowerBound;
            upperBound = initialUpperBound;
            pointDistances = this.#distances.slice(i * this.#n, (i + 1) * this.#n);

            for (let j = 0; j < this.#sigmasMaxIterations; j++) {
                sigma = (upperBound + lowerBound) / 2;
                affinities = this.#getAffinities(pointDistances, sigma, i);
                perplexity = this.#getPerplexity(affinities);
                
                if (Math.abs(perplexity - this.#config.perplexity) <= 1e-6) {
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

        return sigmas;
    }

    #getSymmetricAffinities(): number[][] {
        const affinities = new Array(this.#n);
        const symmetricAffinities = new Array(this.#n);
        let pointDistances: number[];

        for (let i = 0; i < this.#n; i++) {
            pointDistances = this.#distances.slice(i * this.#n, (i + 1) * this.#n);
            affinities[i] = this.#getAffinities(pointDistances, this.#sigmas[i], i);
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

    #getGradient(Y: number[][]): number[] {
        const gradient = new Array(this.#n * this.#config.nDims).fill(0);
        const projectedAffinities = this.#getProjectedAffinities(Y);

        let coef: number;

        for (let i = 0; i < this.#n; i++) {
            for (let j = 0; j < this.#n; j++) {
                coef = this.#affinities[i][j] - projectedAffinities[i * this.#n + j];
                coef *= 1 / (1 + squaredEuclideanDistance(Y[i], Y[j]));

                for (let k = 0; k < this.#config.nDims; k++) {
                    gradient[i * this.#config.nDims + k] += 4 * coef * (Y[i][k] - Y[j][k]);
                }
            }
        }

        return gradient;
    }

    #getInitialSigmaUpperBound(): number {
        const maxDistance = this.#distances.reduce((max, curr) => (curr > max ? curr : max), 0);
        const upperBound = Math.log(maxDistance + 1e-6) + 10;
        return upperBound;
    }

    #getAffinities(pointDistances: number[], sigma: number, rowNo: number): number[] {
        const affinities = new Array(this.#n);
        let sum = 0;

        for (let i = 0; i < this.#n; i++) {
            if (i === rowNo) {
                affinities[i] = 0;
                continue;
            }

            affinities[i] = Math.exp(-pointDistances[i] / (2 * sigma * sigma));
            sum += affinities[i];
        }

        return affinities.map((a) => a / sum);
    }

    #getPerplexity(affinities: number[]): number {
        let H = 0;

        for (const p of affinities) {
            H -= p * Math.log2(p + 1e-10);
        }

        const perplexity = Math.pow(2, H);
        return perplexity;
    }

    #getProjectedAffinities(Y: number[][]): number[] {
        const affinities = new Array(this.#n * this.#n).fill(0);
        let sum = 0;
        let distance: number;
        let affinity: number;

        for (let i = 0; i < this.#n; i++) {
            for (let j = i + 1; j < this.#n; j++) {
                if (i === j) {
                    continue;
                }

                distance = squaredEuclideanDistance(Y[i], Y[j]);
                affinity = 1 / (1 + distance);
                affinities[i * this.#n + j] = affinity;
                affinities[j * this.#n + i] = affinity;
                sum += 2 * affinity;
            }
        }

        for (let i = 0; i < Y.length; i++) {
            for (let j = 0; j < Y.length; j++) {
                affinities[i * this.#n + j] /= sum;
            }
        }

        return affinities;
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
