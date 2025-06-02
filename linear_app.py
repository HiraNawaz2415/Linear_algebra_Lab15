import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import linalg

st.set_page_config(page_title="Linear Algebra Lab", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e2f;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #00c9a7;
    }
    .css-1v3fvcr, .css-ffhzg2 {
        background-color: #2e2e3a !important;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    }
    .stTextArea, .stTextInput, .stSelectbox, .stButton {
        color: #fff !important;
        background-color: #3a3a4f !important;
        border: 1px solid #555 !important;
    }
    .stButton>button {
        color: white;
        background-color: #00c9a7;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #00a98f;
        transform: scale(1.05);
    }
    .stSelectbox>div>div>div {
        color: #e0e0e0;
    }
    .stMarkdown h2 {
        border-bottom: 2px solid #00c9a7;
        padding-bottom: 4px;
    }
    .stTextInput input {
        color: #fff !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📐 Linear Algebra Lab")

st.markdown("""
This app performs a wide range of linear algebra tasks:
- Solve systems of equations (Ax = b)
- Compute matrix inverse, determinant, rank
- Eigenvalues, eigenvectors, and diagonalization
- LU decomposition
- Matrix operations (add, subtract, multiply, transpose)
""")

# Input matrix A
matrix_input = st.text_area("Enter matrix A (rows comma-separated)", "2,1\n1,3")

try:
    A = np.array([list(map(float, row.split(','))) for row in matrix_input.strip().split('\n')])

    with st.expander("📊 Show Matrix A and Properties"):
        st.write("### 🧮 Matrix A")
        st.write(A)
        st.write(f"**Shape:** `{A.shape}`")
        st.write(f"**Rank:** `{np.linalg.matrix_rank(A)}`")
        if A.shape[0] == A.shape[1]:
            det = np.linalg.det(A)
            st.write(f"**Determinant:** `{det:.4f}`")
        else:
            st.info("Matrix is not square — determinant not defined.")

    operation = st.selectbox("Choose an operation", [
        "Solve Ax = b",
        "Matrix Inverse",
        "Determinant",
        "Rank of Matrix",
        "Eigenvalues & Eigenvectors",
        "Diagonalization",
        "LU Decomposition",
        "Matrix Operations"
    ])

    if operation == "Solve Ax = b":
        b_input = st.text_input("Enter vector b (comma-separated)", "3,4")
        b = np.array(list(map(float, b_input.strip().split(','))))
        if A.shape[0] == b.shape[0]:
            try:
                x = linalg.solve(A, b)
                st.success(f"Solution x: {x}")
            except Exception as e:
                st.error(f"Cannot solve: {e}")
        else:
            st.error("Dimensions of A and b do not match.")

    elif operation == "Matrix Inverse":
        if A.shape[0] == A.shape[1]:
            A_inv = linalg.inv(A)
            st.success("Inverse of A:")
            st.write(A_inv)
        else:
            st.error("Matrix must be square.")

    elif operation == "Determinant":
        if A.shape[0] == A.shape[1]:
            det = linalg.det(A)
            st.success(f"Determinant: {det:.4f}")
        else:
            st.error("Matrix must be square.")

    elif operation == "Rank of Matrix":
        rank = np.linalg.matrix_rank(A)
        st.success(f"Rank: {rank}")

    elif operation == "Eigenvalues & Eigenvectors":
        if A.shape[0] == A.shape[1]:
            eigvals, eigvecs = linalg.eig(A)
            with st.expander("📈 Eigenvalues and Eigenvectors"):
                st.success("Eigenvalues:")
                st.write(np.round(eigvals, 4))
                st.success("Eigenvectors (columns):")
                st.write(np.round(eigvecs, 4))

                # Heatmap of eigenvectors
                fig, ax = plt.subplots()
                sns.heatmap(np.abs(eigvecs), annot=True, cmap="viridis", ax=ax)
                ax.set_title("Heatmap of |Eigenvectors|")
                st.pyplot(fig)
        else:
            st.error("Matrix must be square.")

    elif operation == "Diagonalization":
        if A.shape[0] == A.shape[1]:
            eigvals, eigvecs = linalg.eig(A)
            try:
                P_inv = np.linalg.inv(eigvecs)
                D = np.diag(eigvals)
                A_diag = eigvecs @ D @ P_inv
                with st.expander("🧾 Diagonalization Output"):
                    st.success("Matrix is diagonalizable.")

                    st.write("**P (eigenvectors):**")
                    st.write(np.round(eigvecs, 4))

                    st.write("**D (diagonal matrix of eigenvalues):**")
                    st.write(np.round(D, 4))

                    # Heatmap of D
                    fig, ax = plt.subplots()
                    sns.heatmap(np.abs(D), annot=True, cmap="magma", ax=ax)
                    ax.set_title("Heatmap of Diagonal Matrix D")
                    st.pyplot(fig)

                    st.write("**A ≈ P·D·P⁻¹:**")
                    st.write(np.round(A_diag, 4))
            except:
                st.error("Matrix is not diagonalizable.")
        else:
            st.error("Matrix must be square.")

    elif operation == "LU Decomposition":
        if A.shape[0] == A.shape[1]:
            P, L, U = linalg.lu(A)
            with st.expander("📚 LU Decomposition Result"):
                st.success("P matrix:")
                st.write(P)
                st.success("L matrix:")
                st.write(L)
                fig, ax = plt.subplots()
                sns.heatmap(np.abs(L), annot=True, cmap="Blues", ax=ax)
                ax.set_title("Heatmap of L Matrix")
                st.pyplot(fig)

                st.success("U matrix:")
                st.write(U)
                fig, ax = plt.subplots()
                sns.heatmap(np.abs(U), annot=True, cmap="Reds", ax=ax)
                ax.set_title("Heatmap of U Matrix")
                st.pyplot(fig)
        else:
            st.error("Matrix must be square.")

    elif operation == "Matrix Operations":
        matrix_b_input = st.text_area("Enter matrix B (for operations)", "1,0\n0,1")
        try:
            B = np.array([list(map(float, row.split(','))) for row in matrix_b_input.strip().split('\n')])
            st.write("Matrix B:")
            st.write(B)

            op = st.selectbox("Select matrix operation", ["A + B", "A - B", "A x B", "Aᵀ", "Bᵀ"])

            if op == "A + B":
                if A.shape == B.shape:
                    st.write("A + B =")
                    st.write(A + B)
                else:
                    st.error("Shape mismatch for addition.")

            elif op == "A - B":
                if A.shape == B.shape:
                    st.write("A - B =")
                    st.write(A - B)
                else:
                    st.error("Shape mismatch for subtraction.")

            elif op == "A x B":
                try:
                    product = A @ B
                    st.write("A × B =")
                    st.write(product)
                except:
                    st.error("Shape mismatch for multiplication.")

            elif op == "Aᵀ":
                st.write("Transpose of A:")
                st.write(A.T)

            elif op == "Bᵀ":
                st.write("Transpose of B:")
                st.write(B.T)

        except Exception as e:
            st.error(f"Error parsing Matrix B: {e}")

except Exception as e:
    st.error(f"Matrix A input error: {e}")
