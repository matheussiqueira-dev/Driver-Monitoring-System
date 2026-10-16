/** Saves text as a local file through a temporary object URL; nothing is uploaded. */
export function downloadText(filename: string, content: string, mimeType: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mimeType }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.rel = "noopener";
  document.body.append(link);
  link.click();
  link.remove();
  // Revoke after the click has been handled so the download is not cancelled.
  setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 0);
}
