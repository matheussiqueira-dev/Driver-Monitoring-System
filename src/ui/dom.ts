import { AppError } from "../shared/errors";

/** querySelector that fails fast with a typed AppError instead of returning null. */
export function requireElement<T extends Element>(selector: string, type: new () => T, root: ParentNode = document): T {
  const element = root.querySelector(selector);
  if (!(element instanceof type)) {
    throw new AppError("DOM_ELEMENT_MISSING", `Elemento obrigatorio ausente ou com tipo inesperado: ${selector}`);
  }
  return element;
}

export function requireAll<T extends Element>(selector: string, type: new () => T, root: ParentNode = document): T[] {
  const elements = [...root.querySelectorAll(selector)].filter((element): element is T => element instanceof type);
  if (elements.length === 0) {
    throw new AppError("DOM_ELEMENT_MISSING", `Nenhum elemento encontrado para: ${selector}`);
  }
  return elements;
}

/** Only touch the DOM when the text actually changes (avoids layout work at 60 fps). */
export function setText(element: Element, text: string): void {
  if (element.textContent !== text) {
    element.textContent = text;
  }
}
