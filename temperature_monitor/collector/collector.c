#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <time.h>


#define BUFFER_SIZE 13

int main(int argc, char *argv[]) {
	if (argc != 4) {
		fprintf(stderr, "Usage: %s <server_address> <port> <output_file>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	char *server_address = argv[1];
	char port = atoi(argv[2]);
	char *output_file = argv[3];
	int sock = 0;
	struct sockaddr_in serv_addr;
	char buffer[BUFFER_SIZE] = {0};
	FILE *file;

	// Create socket file descriptor
	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		printf("Socket creation error");
		exit(EXIT_FAILURE);
	}

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(port);

	// Convert IPv4 and IPv6 addresses from text to binary form
	if (inet_pton(AF_INET, server_address, &serv_addr.sin_addr) <= 0) {
		printf("Invalid address or address not supported");
		close(sock);
		exit(EXIT_FAILURE);
	}

	// Connect to the server
	if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
		printf("Connection failed");
		close(sock);
		exit(EXIT_FAILURE);
	}

	printf("Connected to %s on port %d\n", server_address, port);

	// Open file for appending
	file = fopen(output_file, "a");
	if (file == NULL) {
		printf("Failed to open file");
		close(sock);
		exit(EXIT_FAILURE);
	}

	// Listen for data from the server
	int read_size;
	read_size = read(sock, buffer, BUFFER_SIZE - 1);
	buffer[read_size] = '\0';  // Null-terminate the buffer
	printf("Received: %s\n", buffer);
	fprintf(file, "%lu,%s",(unsigned long)time(NULL), buffer);
	fflush(file);  // Ensure the data is written to the file immediately


	if (read_size < 0) {
		printf("Read error");
	}

	// Close the file and the socket
	fclose(file);
	close(sock);

	return 0;
}
