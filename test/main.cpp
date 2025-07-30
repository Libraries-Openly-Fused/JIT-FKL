/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <iostream>

extern int launcher();

int main(int argc, char **argv) {
    try {
        int result = launcher();
        if (result == 0) {
            std::cout << "Test PASSED" << std::endl;
        } else {
            std::cout << "Test FAILED with code: " << result << std::endl;
        }
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test FAILED with unknown exception" << std::endl;
        return 1;
    }
}