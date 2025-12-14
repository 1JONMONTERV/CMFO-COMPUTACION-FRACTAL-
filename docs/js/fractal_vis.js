// CMFO Fractal Attractor Visualizer
// Renders the 7D attractor projection on canvas

const canvas = document.getElementById('fractal-canvas');
const ctx = canvas.getContext('2d');

let width, height;
let particles = [];
const PARTICLE_COUNT = 100;
const PHI = 1.6180339887;

function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
}

class Particle {
    constructor() {
        this.reset();
    }

    reset() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.state = Math.random(); // 1D projection of T7 state
        this.size = Math.random() * 2 + 1;
    }

    update() {
        // Fractal attraction logic (Geometric Pull)
        const centerX = width / 2;
        const centerY = height / 2;

        // Attraction to center controlled by PHI
        const dx = centerX - this.x;
        const dy = centerY - this.y;

        this.vx += dx * 0.0001 * this.state;
        this.vy += dy * 0.0001 * this.state;

        // Apply velocity
        this.x += this.vx;
        this.y += this.vy;

        // Friction
        this.vx *= 0.99;
        this.vy *= 0.99;

        // Update Internal State (Rotation)
        this.state = (this.state + PHI) % 1.0;

        // Reset if out of bounds (rare due to attraction)
        if (this.x < 0 || this.x > width || this.y < 0 || this.y > height) {
            // this.reset();
        }
    }

    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);

        // Color based on velocity
        const speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
        const alpha = Math.min(1, speed * 0.5);
        // Mix Gold and Cyan based on state
        const r = this.state > 0.5 ? 212 : 0;
        const g = this.state > 0.5 ? 175 : 240;
        const b = this.state > 0.5 ? 55 : 255;
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        ctx.fill();

        // Connect to neighbors
        particles.forEach(p => {
            const d = Math.sqrt((p.x - this.x) ** 2 + (p.y - this.y) ** 2);
            if (d < 100) {
                ctx.beginPath();
                ctx.moveTo(this.x, this.y);
                ctx.lineTo(p.x, p.y);
                ctx.strokeStyle = `rgba(212, 175, 55, ${0.15 * (1 - d / 100)})`;
                ctx.stroke();
            }
        });
    }
}

function init() {
    resize();
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push(new Particle());
    }
    animate();
}

function animate() {
    ctx.clearRect(0, 0, width, height);

    particles.forEach(p => {
        p.update();
        p.draw();
    });

    requestAnimationFrame(animate);
}

window.addEventListener('resize', resize);
window.addEventListener('DOMContentLoaded', init);
