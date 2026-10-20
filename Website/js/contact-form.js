// The contact form: checks each field, sends the message through the form
// service in the form's action (Formspree), then shows a confirmation dialog.
// Without JavaScript the form still posts normally to the same address.
(() => {
  'use strict';
  const M = (window.R26SITE = window.R26SITE || {});
  const { $ } = M;

  const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  function initContactForm() {
    const form = $('#contact-form');
    const dialog = $('#contact-confirm');
    if (!form || !dialog) return;
    form.noValidate = true; // our own messages replace the browser's bubbles
    const send = $('.contact-form__send', form);
    const title = $('.confirm__title', dialog);
    const text = $('.confirm__text', dialog);

    const fields = {
      name: (v) => (v.trim() ? '' : 'Please enter your name.'),
      email: (v) => (!v.trim() ? 'Please enter your email address.' : EMAIL.test(v.trim()) ? '' : 'Please enter a valid email address.'),
      subject: (v) => (v.trim() ? '' : 'Please enter a subject.'),
      message: (v) => (v.trim().length >= 10 ? '' : 'Please write a message of at least 10 characters.'),
    };
    const check = (name) => {
      const input = form.elements[name];
      const msg = fields[name](input.value);
      const err = $(`#${input.id}-err`);
      err.textContent = msg;
      if (msg) {
        input.setAttribute('aria-invalid', 'true');
        input.setAttribute('aria-describedby', err.id);
      } else {
        input.removeAttribute('aria-invalid');
        input.removeAttribute('aria-describedby');
      }
      return !msg;
    };
    // Re-check a field as soon as it is corrected.
    Object.keys(fields).forEach((name) => {
      form.elements[name].addEventListener('input', () => {
        if (form.elements[name].hasAttribute('aria-invalid')) check(name);
      });
    });

    const open = (kind, heading, body) => {
      dialog.dataset.kind = kind;
      title.textContent = heading;
      text.textContent = body;
      if (typeof dialog.showModal === 'function') dialog.showModal();
      else window.alert(`${heading}\n\n${body}`);
    };
    dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); }); // click outside closes

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const ok = Object.keys(fields).map(check).every(Boolean);
      if (!ok) {
        form.querySelector('[aria-invalid="true"]').focus();
        return;
      }
      const name = form.elements.name.value.trim().split(/\s+/)[0];
      const email = form.elements.email.value.trim();

      // No form service yet ([FORM_ID] still in the action): hand the message to
      // the visitor's own email app, addressed to the team, so nothing is lost.
      if ((form.getAttribute('action') || '').includes('[FORM_ID]')) {
        const to = form.dataset.mailto;
        const subject = `[R26-CS-010] ${form.elements.subject.value.trim()}`;
        const body = `${form.elements.message.value.trim()}\n\n${form.elements.name.value.trim()} (${email})`;
        window.location.href = `mailto:${to}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
        open('ok', 'Almost done', `Thank you, ${name}. Your email app has opened with the message ready; press send there to deliver it.`);
        return;
      }

      send.disabled = true;
      send.textContent = 'Sending';
      try {
        const res = await fetch(form.action, { method: 'POST', body: new FormData(form), headers: { Accept: 'application/json' } });
        if (!res.ok) throw new Error(String(res.status));
        form.reset();
        open('ok', 'Message sent', `Thank you, ${name}. Your message was received, and the team will reply to ${email}.`);
      } catch (err) {
        open('error', 'Message not sent', 'Something went wrong while sending. Please try again in a moment.');
      } finally {
        send.disabled = false;
        send.textContent = 'Send message';
      }
    });
  }

  Object.assign(M, { initContactForm });
})();
