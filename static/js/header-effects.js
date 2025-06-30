// Script pour améliorer l'effet sticky du header
window.addEventListener('scroll', function() {
    const header = document.querySelector('.header');
    if (header) {
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    }
});

// Fonction d'initialisation pour les pages qui en ont besoin
function initHeaderEffects() {
    // Ajouter d'autres effets header si nécessaire dans le futur
    console.log('Header effects initialized');
}

// Auto-initialisation
document.addEventListener('DOMContentLoaded', function() {
    initHeaderEffects();
});
