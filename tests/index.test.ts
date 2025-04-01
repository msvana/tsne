import { expect, test } from "vitest";
import { TSNE, squaredEuclideanDistance } from "../src/index";

async function getEmbeddings(texts: string[]): Promise<number[][]> {
    const response = await fetch("https://embeddings.swarm.svana.name/embeddings", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            inputs: texts,
        }),
    });

    const result = await response.json();
    return result.embeddings;
}

test("letter embeddings", async () => {
    const texts = ["A", "A", "B", "C", "C", "D"];
    const embeddings = await getEmbeddings(texts);

    const tsne = new TSNE({ perplexity: Math.floor(texts.length / 2) });
    const projections = tsne.transform(embeddings);
    const [A1, A2, B, C1, C2, D] = projections;

    expect(squaredEuclideanDistance(A1, A2)).toBeLessThan(1e-5);
    expect(squaredEuclideanDistance(C1, C2)).toBeLessThan(1e-5);

    expect(squaredEuclideanDistance(A1, B)).toBeGreaterThan(1e-3);
    expect(squaredEuclideanDistance(C1, D)).toBeGreaterThan(1e-3);
    expect(squaredEuclideanDistance(B, D)).toBeGreaterThan(1e-3);
});

test("sentiment embeddings", async () => {
    const texts = [
        "This is great news! I am so happy!",
        "Very positive",
        "Positive",
        "Neutral",
        "Negative",
        "Very Negative",
    ];

    const embeddings = await getEmbeddings(texts);

    const tsne = new TSNE({ perplexity: Math.floor(texts.length / 2) });
    const projections = tsne.transform(embeddings);
    const [SENT, VPOS, POS, NEUT, NEG, VNEG] = projections;

    const distVPOS = squaredEuclideanDistance(SENT, VPOS);
    const distPOS = squaredEuclideanDistance(SENT, POS);
    const distNEUT = squaredEuclideanDistance(SENT, NEUT);
    const distNEG = squaredEuclideanDistance(SENT, NEG);
    const distVNEG = squaredEuclideanDistance(SENT, VNEG);

    expect(distVPOS).toBeLessThan(distNEUT);
    expect(distPOS).toBeLessThan(distNEUT);
    expect(distPOS).toBeLessThan(distNEG);
    expect(distPOS).toBeLessThan(distVNEG);
});
