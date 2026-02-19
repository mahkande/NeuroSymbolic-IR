param(
  [string]$Text = "",
  [string]$Source = "cursor",
  [string]$File = "conversation.txt",
  [switch]$Clipboard
)

if (-not $Text -and $Clipboard) {
  $Text = Get-Clipboard
}

if (-not $Text) {
  Write-Error "No text provided. Use -Text or -Clipboard."
  exit 1
}

$ts = (Get-Date).ToUniversalTime().ToString("o")
$block = "[$ts] source=$Source`n$Text`n`n"
Add-Content -Path $File -Value $block -Encoding UTF8
Write-Output "Appended to $File"
