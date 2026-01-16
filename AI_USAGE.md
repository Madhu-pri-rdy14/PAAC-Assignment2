# AI Attribution

## 1. AI Tools Used
I used **Gemini** (Google) as an AI thought partner and coding assistant for this assignment.

## 2. Scope of Assistance
The AI was used for the following specific tasks:
* **Code Structure:** Helping to organize the project into `src/` folders (data, model, train) and setting up the standard PyTorch boilerplate.
* **Debugging:** Assisting in resolving dimension mismatch errors during the implementation of the Cholesky decomposition layer.
* **Documentation:** Drafting the initial outline for `README.md` and the replication guide in `docs/report.md`.

## 3. Human Verification
I certify that I have manually verified the correctness of the code and results, specifically:
* **Physical Constraints:** I manually checked the mathematical logic of the formula $\rho = L L^\dagger / \text{Tr}(L L^\dagger)$ to ensure it produces valid density matrices.
* **Metrics:** I independently ran the inference script to confirm the Latency and Fidelity numbers reported in the documentation.