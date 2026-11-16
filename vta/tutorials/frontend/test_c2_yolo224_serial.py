"""UART mutation must follow a real matching board identity, never a prompt alone."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from c2_yolo224_serial import recover

MAC='00:0a:35:00:22:01';CID='ff56784d4341524400a00063d9000100'
BOOT='11111111-2222-3333-4444-555555555555';PROMPT='root@axu:~# '


def identity(mac=MAC,cid=CID):
    return 'C2_SERIAL_ID_BEGIN\n'+BOOT+'\n'+mac+'\n'+cid+'\nC2_SERIAL_ID_END\n'+PROMPT


class SerialIdentityTests(unittest.TestCase):
    def test_matching_board_can_restore_only_transient_IP(self):
        with tempfile.TemporaryDirectory() as temp,patch('c2_yolo224_serial.exchange',
                side_effect=[PROMPT,identity(),'C2_TRANSIENT_NETWORK_RESTORED\n'+PROMPT]) as exchange:
            out=Path(temp);r=recover(out,'COM3','192.168.1.247',MAC,CID)
            self.assertEqual(r['boot_id'],BOOT);self.assertFalse(r['persistent_configuration_modified'])
            self.assertIn('ifconfig eth0 192.168.1.247',exchange.call_args_list[-1].args[2])
            self.assertTrue((out/'serial_network_recovery.json').exists())

    def test_wrong_MAC_or_SD_never_receives_ifconfig(self):
        for response in (identity(mac='00:00:00:00:00:01'),identity(cid='0'*32)):
            with self.subTest(response=response),tempfile.TemporaryDirectory() as temp,patch(
                    'c2_yolo224_serial.exchange',side_effect=[PROMPT,response]) as exchange:
                with self.assertRaisesRegex(ValueError,'unverified board'):recover(Path(temp),'COM3','192.168.1.247',MAC,CID)
                self.assertEqual(exchange.call_count,2)
                self.assertTrue(all('ifconfig' not in c.args[2] for c in exchange.call_args_list))

    def test_password_prompt_does_not_trigger_password_attempt(self):
        with tempfile.TemporaryDirectory() as temp,patch('c2_yolo224_serial.exchange',
                side_effect=['axu login: ','Password: ']) as exchange:
            with self.assertRaisesRegex(RuntimeError,'root shell'):recover(Path(temp),'COM3','192.168.1.247',MAC,CID)
            self.assertEqual([c.args[2] for c in exchange.call_args_list],['\r','root\r'])

    def test_bootloader_prompt_requires_user(self):
        with tempfile.TemporaryDirectory() as temp,patch('c2_yolo224_serial.exchange',return_value='ZynqMP> ') as exchange:
            with self.assertRaisesRegex(RuntimeError,'root shell'):recover(Path(temp),'COM3','192.168.1.247',MAC,CID)
            self.assertEqual(exchange.call_count,1)


if __name__=='__main__':unittest.main()
