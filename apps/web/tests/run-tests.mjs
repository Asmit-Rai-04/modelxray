import { spawnSync } from 'node:child_process';
import process from 'node:process';

// Keep the user's app on ports 3000/8000 and its evidence store untouched.
const env = { ...process.env, NEXT_PUBLIC_API_URL: 'http://localhost:8001' };
function run(command, args) {
  const result = spawnSync(command, args, { env, stdio: 'inherit', shell: process.platform === 'win32' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
run('python', ['-X', 'utf8', 'tests/prepare_fixtures.py']);
run('npm', ['run', 'typecheck']);
run('npm', ['run', 'build']);
run('npx', ['playwright', 'test', ...process.argv.slice(2)]);
