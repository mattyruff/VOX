# Minimal static file server for local preview/testing of the Vox app.
# Serves the folder this script lives in on http://localhost:8137/
$root = $PSScriptRoot
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://localhost:8137/")
$listener.Start()
Write-Host "Vox dev server: http://localhost:8137/  (root: $root)"
while ($listener.IsListening) {
  try {
    $ctx = $listener.GetContext()
    $path = [System.Uri]::UnescapeDataString($ctx.Request.Url.LocalPath).TrimStart('/')
    if ([string]::IsNullOrEmpty($path)) { $path = "index.html" }
    $file = Join-Path $root $path
    if (Test-Path $file -PathType Leaf) {
      $bytes = [System.IO.File]::ReadAllBytes($file)
      $ext = [System.IO.Path]::GetExtension($file).ToLower()
      switch ($ext) {
        ".html" { $ctx.Response.ContentType = "text/html; charset=utf-8" }
        ".js"   { $ctx.Response.ContentType = "application/javascript; charset=utf-8" }
        ".css"  { $ctx.Response.ContentType = "text/css; charset=utf-8" }
        default { $ctx.Response.ContentType = "application/octet-stream" }
      }
      $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    } else {
      $ctx.Response.StatusCode = 404
    }
    $ctx.Response.Close()
  } catch {}
}
