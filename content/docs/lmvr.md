---
weight: 100
title: "Large Language Models via Rust"
description: "The State of the Art Open Foundational Models in Rust"
icon: "menu_book"
date: "2024-11-19T16:42:29.189613+07:00"
lastmod: "2024-11-19T16:42:29.189613+07:00"
draft: false
toc: true
---

{{< figure src="/images/cover.png" width="500" height="300" class="text-center" >}}

<center>

## 📘 About This Book

</center>

{{% alert icon="📘" context="info" %}}
<p style="text-align: justify;">
"LMVR - Large Language Models via Rust" is a pioneering open-source project that bridges the power of foundational models with the robustness of the Rust programming language. It highlights Rust's strengths in performance, safety, and concurrency while advancing the state-of-the-art in AI. Tailored for students, researchers, and professionals, LMVR delivers a comprehensive guide to building scalable, efficient, and secure large language models. By leveraging Rust, this book ensures that cutting-edge research and practical solutions go hand-in-hand. Readers will gain in-depth knowledge of model architectures, training methodologies, and real-world deployments, all while mastering Rust's unique capabilities for AI development.
</p>
{{% /alert %}}

<div class="row justify-content-center my-4">
    <div class="col-md-8 col-12">
        <div class="card p-4 text-center support-card">
            <h4 class="mb-3" style="color: #00A3C4;">SUPPORT US ❤️</h4>
            <p class="card-text">
                Support our mission by purchasing or sharing the LMVR companion guide.
            </p>
            <div class="d-flex justify-content-center mb-3 flex-wrap">
                <a href="https://www.amazon.com/dp/B0DK2NH9CZ" class="btn btn-lg btn-outline-support m-2 support-btn">
                    <img src="../../images/kindle.png" alt="Amazon Logo" class="support-logo-image">
                    <span class="support-btn-text">Buy on Amazon</span>
                </a>
                <a href="https://play.google.com/store/books/details?id=NnwpEQAAQBAJ" class="btn btn-lg btn-outline-support m-2 support-btn">
                    <img src="../../images/GBooks.png" alt="Google Books Logo" class="support-logo-image">
                    <span class="support-btn-text">Buy on Google Books</span>
                </a>
            </div>
        </div>
    </div>
</div>

<style>
    .btn-outline-support {
        color: #00A3C4;
        border: 2px solid #00A3C4;
        background-color: transparent;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 25px;
        width: 200px;
        text-align: center;
        transition: all 0.3s ease-in-out;
    }
    .btn-outline-support:hover {
        background-color: #00A3C4;
        color: white;
        border-color: #00A3C4;
    }
    .support-logo-image {
        max-width: 100%;
        height: auto;
        margin-bottom: 16px;
    }
    .support-btn {
        width: 300px;
    }
    .support-btn-text {
        font-weight: bold;
        font-size: 1.1rem;
    }
    .support-card {
        transition: box-shadow 0.3s ease-in-out;
    }
    .support-card:hover {
        box-shadow: 0 0 20px #00A3C4;
    }
</style>

<center>

## 🚀 About RantAI

</center>

<div class="row justify-content-center">
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://rantai.dev/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="/images/Logo.png" class="card-img-top" alt="Rantai Logo">
            </div>
        </a>
    </div>
</div>

{{% alert icon="🚀" context="success" %}}
<p style="text-align: justify;">
RantAI is a dynamic Indonesian tech startup dedicated to leveraging Rust for AI-driven solutions. As a premier System Integrator (SI), RantAI excels in Machine Learning, Deep Learning, and Digital Twin simulations, delivering solutions for scientific and industrial challenges. Through LMVR, RantAI empowers the global AI community to adopt Rust, ensuring innovation is both scalable and ethical.
</p>
{{% /alert %}}

<center>

## 👥 LMVR Authors

</center>
<div class="row flex-xl-wrap pb-4">
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://www.linkedin.com/in/shirologic/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-1EMgqgjvaVvYZ7wbZ7Zm-v1.png" class="card-img-top" alt="Evan Pradipta Hardinatha">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Evan Pradipta Hardinatha</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://www.linkedin.com/in/jaisy-arasy/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-cHU7kr5izPad2OAh1eQO-v1.png" class="card-img-top" alt="Jaisy Malikulmulki Arasy">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Jaisy Malikulmulki Arasy</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://www.linkedin.com/in/chevhan-walidain/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-UTFiCKrYqaocqib3YNnZ-v1.png" class="card-img-top" alt="Chevan Walidain">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Chevan Walidain</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://www.linkedin.com/in/idham-multazam/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-Ra9qnq6ahPYHkvvzi71z-v1.png" class="card-img-top" alt="Idham Hanif Multazam">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Idham Hanif Multazam</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="http://www.linkedin.com">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-0n0SFhW3vVnO5VXX9cIX-v1.png" class="card-img-top" alt="Razka Athallah Adnan">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Razka Athallah Adnan</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="http://linkedin.com">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-vto2jpzeQkntjXGi2Wbu-v1.png" class="card-img-top" alt="Raffy Aulia Adnan">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Raffy Aulia Adnan</p>
                </div>
            </div>
        </a>
    </div>
</div>
