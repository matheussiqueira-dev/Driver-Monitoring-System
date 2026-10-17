export type AppErrorCode =
  | "CONFIG_INVALID"
  | "DOM_ELEMENT_MISSING"
  | "CAMERA_UNSUPPORTED"
  | "CAMERA_PERMISSION_DENIED"
  | "CAMERA_UNAVAILABLE"
  | "MODEL_LOAD_FAILED"
  | "UNKNOWN";

/** Error with a stable code, so UI and telemetry can react without parsing messages. */
export class AppError extends Error {
  readonly code: AppErrorCode;

  constructor(code: AppErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "AppError";
    this.code = code;
  }
}

/** Normalize anything thrown (DOMException, string, unknown) into an AppError. */
export function toAppError(error: unknown, fallback: AppErrorCode = "UNKNOWN"): AppError {
  if (error instanceof AppError) {
    return error;
  }
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return new AppError("CAMERA_PERMISSION_DENIED", "Permissao de camera negada", { cause: error });
    }
    if (error.name === "NotFoundError" || error.name === "NotReadableError" || error.name === "OverconstrainedError") {
      return new AppError("CAMERA_UNAVAILABLE", "Nenhuma camera disponivel ou ela esta em uso", { cause: error });
    }
  }
  const message = error instanceof Error ? error.message : String(error);
  return new AppError(fallback, message || "Erro desconhecido", { cause: error });
}

const USER_MESSAGES: Record<AppErrorCode, string> = {
  CONFIG_INVALID: "Configuracao invalida. Recarregue a pagina.",
  DOM_ELEMENT_MISSING: "Falha ao montar a interface. Recarregue a pagina.",
  CAMERA_UNSUPPORTED: "Este navegador nao suporta acesso a camera.",
  CAMERA_PERMISSION_DENIED: "Permissao de camera negada. Libere o acesso nas configuracoes do navegador.",
  CAMERA_UNAVAILABLE: "Nenhuma camera disponivel, ou ela esta em uso por outro app.",
  MODEL_LOAD_FAILED: "Nao foi possivel carregar o modelo de visao. Verifique a conexao.",
  UNKNOWN: "Erro inesperado. A demo segue ativa.",
};

export function userMessageFor(error: AppError): string {
  return USER_MESSAGES[error.code];
}
