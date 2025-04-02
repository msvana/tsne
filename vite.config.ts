import { defineConfig } from "vite";

const removeCrossOrigin = () => {
    return {
        name: "remove-crossorigin-attribute",
        transformIndexHtml(html: string) {
            return html
                .replace(/crossorigin /g, "")
                .replace('<script type="module" ', "<script defer ")
                .replace("/assets", "assets");
        },
    };
};

export default defineConfig({
    root: "src/",
    build: {
        rollupOptions: {
            input: {
                main: "src/example.html",
            },
        },
        outDir: "../demo/",
    },
    plugins: [removeCrossOrigin()],
});
