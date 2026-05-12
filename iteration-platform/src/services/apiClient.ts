import { API_BASE_URL } from '../config/env';

type ApiEnvelope<T> = {
  success: boolean;
  data?: T;
  error?: string;
};

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

type QueryParams = Record<string, string | number | boolean | undefined>;

function buildUrl(path: string, params?: QueryParams): string {
  const normalizedBase = API_BASE_URL.replace(/\/$/, '');
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  const url = new URL(`${normalizedBase}${normalizedPath}`);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        url.searchParams.set(key, String(value));
      }
    });
  }
  return url.toString();
}

async function parseResponse<T>(response: Response): Promise<T> {
  const text = await response.text();
  const payload = text ? (JSON.parse(text) as ApiEnvelope<T>) : undefined;

  if (!response.ok) {
    throw new ApiError(response.status, payload?.error || response.statusText || 'API request failed');
  }

  if (!payload) {
    throw new ApiError(response.status, 'Empty API response');
  }

  if (!payload.success) {
    throw new ApiError(response.status, payload.error || 'API request failed');
  }

  if (payload.data === undefined) {
    throw new ApiError(response.status, 'API response missing data');
  }

  return payload.data;
}

export async function get<T>(path: string, params?: QueryParams): Promise<T> {
  const response = await fetch(buildUrl(path, params), {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  });
  return parseResponse<T>(response);
}

export async function post<TReq, TResp>(path: string, body?: TReq): Promise<TResp> {
  const response = await fetch(buildUrl(path), {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return parseResponse<TResp>(response);
}
