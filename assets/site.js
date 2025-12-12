
(function(){
  const path = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
  document.querySelectorAll('[data-nav]').forEach(a=>{
    const href = (a.getAttribute('href')||'').split('/').pop().toLowerCase();
    if(href === path){ a.classList.add('active'); }
  });
})();
