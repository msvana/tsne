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
    #sigmasMaxIterations: number = 50;

    constructor(config: TSNEConfig = {}) {
        this.#config = { ...defaultConfig, ...config };
    }

    transform(X: number[][]): number[][] {
        this.#validateInput(X);
        this.#n = X.length;
        this.#distances = this.#getPairwiseDistances(X);
        this.#sigmas = this.#findBestSigmas();

        let momentum;
        let Y = this.#initRandomProjection();
        let YPrev = structuredClone(Y);
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

                distance = this.#squaredEuclideanDistance(X[i], X[j]);
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

    #initRandomProjection(): number[][] {
        const Y: number[][] = new Array(this.#n);

        for (let i = 0; i < this.#n, i++) {
            Y[i] = new Array(this.#config.nDims);

            for(let j = 0; j < this.#config.nDims; j++) {
                Y[i][j] = Math.random() - 0.5
            }
        }

        return Y;
    }

    #squaredEuclideanDistance(a: number[], b: number[]): number {
        let sum: number = 0;
        let dimDifference: number;

        for (let i = 0; i < a.length; i++) {
            dimDifference = a[i] - b[i];
            sum += dimDifference * dimDifference;
        }

        return sum;
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
}
